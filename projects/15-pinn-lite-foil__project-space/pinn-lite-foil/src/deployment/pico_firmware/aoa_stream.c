/**
 * PINN-Lite-Foil: Raspberry Pi Pico AoA Streaming Firmware
 * 
 * Reads angle-of-attack (pitch/roll) from MPU6050 IMU and streams
 * data via UART for real-time PINN inference.
 *
 * Hardware:
 *   - Raspberry Pi Pico (RP2040)
 *   - MPU6050 IMU (I2C: SDA=GPIO4, SCL=GPIO5)
 *   - Optional: Pitot tube (MS4525DO) on I2C
 *
 * Output Format (UART 115200 baud):
 *   "AoA,<pitch>,<roll>,<airspeed>,<timestamp>\n"
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DPICO_SDK_PATH=/path/to/pico-sdk
 *   make -j$(nproc)
 *
 * Flash:
 *   picotool load aoa_stream.uf2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pico/stdlib.h"
#include "pico/time.h"
#include "hardware/i2c.h"
#include "hardware/gpio.h"
#include "hardware/uart.h"

// ============================================================================
// Configuration
// ============================================================================

// I2C Configuration
#define I2C_PORT        i2c0
#define I2C_SDA_PIN     4
#define I2C_SCL_PIN     5
#define I2C_FREQ_HZ     400000  // 400 kHz

// UART Configuration
#define UART_ID         uart0
#define UART_TX_PIN     0
#define UART_RX_PIN     1
#define UART_BAUD       115200

// Sampling Configuration
#define SAMPLE_RATE_HZ  20      // 20 Hz = 50ms period
#define SAMPLE_PERIOD_MS (1000 / SAMPLE_RATE_HZ)

// MPU6050 I2C Address
#define MPU6050_ADDR    0x68

// MPU6050 Registers
#define MPU6050_REG_PWR_MGMT_1      0x6B
#define MPU6050_REG_ACCEL_XOUT_H    0x3B
#define MPU6050_REG_GYRO_XOUT_H     0x43
#define MPU6050_REG_WHO_AM_I        0x75
#define MPU6050_REG_CONFIG          0x1A
#define MPU6050_REG_ACCEL_CONFIG    0x1C
#define MPU6050_REG_GYRO_CONFIG     0x1B

// MS4525DO Pitot Tube Address (optional)
#define MS4525DO_ADDR   0x28

// ============================================================================
// MPU6050 Driver
// ============================================================================

typedef struct {
    float accel_x, accel_y, accel_z;  // m/s^2
    float gyro_x, gyro_y, gyro_z;      // deg/s
    float pitch, roll;                  // degrees
} mpu6050_data_t;

static bool mpu6050_initialized = false;

/**
 * Write single byte to MPU6050 register
 */
static int mpu6050_write_reg(uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    return i2c_write_blocking(I2C_PORT, MPU6050_ADDR, buf, 2, false);
}

/**
 * Read bytes from MPU6050 starting at register
 */
static int mpu6050_read_regs(uint8_t reg, uint8_t *buf, size_t len) {
    i2c_write_blocking(I2C_PORT, MPU6050_ADDR, &reg, 1, true);
    return i2c_read_blocking(I2C_PORT, MPU6050_ADDR, buf, len, false);
}

/**
 * Initialize MPU6050
 */
bool mpu6050_init(void) {
    // Check WHO_AM_I register
    uint8_t who_am_i;
    if (mpu6050_read_regs(MPU6050_REG_WHO_AM_I, &who_am_i, 1) < 0) {
        printf("[MPU6050] I2C read failed\n");
        return false;
    }
    
    if (who_am_i != 0x68) {
        printf("[MPU6050] Unexpected WHO_AM_I: 0x%02X\n", who_am_i);
        return false;
    }
    
    // Wake up (clear sleep bit)
    mpu6050_write_reg(MPU6050_REG_PWR_MGMT_1, 0x00);
    sleep_ms(100);
    
    // Set DLPF (Digital Low Pass Filter) to ~20Hz bandwidth
    mpu6050_write_reg(MPU6050_REG_CONFIG, 0x04);
    
    // Accelerometer: ±4g range
    mpu6050_write_reg(MPU6050_REG_ACCEL_CONFIG, 0x08);
    
    // Gyroscope: ±500 deg/s range
    mpu6050_write_reg(MPU6050_REG_GYRO_CONFIG, 0x08);
    
    mpu6050_initialized = true;
    printf("[MPU6050] Initialized successfully\n");
    
    return true;
}

/**
 * Read and compute pitch/roll from MPU6050
 */
bool mpu6050_read(mpu6050_data_t *data) {
    if (!mpu6050_initialized) {
        return false;
    }
    
    // Read 14 bytes: Accel (6) + Temp (2) + Gyro (6)
    uint8_t buf[14];
    if (mpu6050_read_regs(MPU6050_REG_ACCEL_XOUT_H, buf, 14) < 0) {
        return false;
    }
    
    // Parse accelerometer (±4g, sensitivity = 8192 LSB/g)
    int16_t accel_x_raw = (buf[0] << 8) | buf[1];
    int16_t accel_y_raw = (buf[2] << 8) | buf[3];
    int16_t accel_z_raw = (buf[4] << 8) | buf[5];
    
    data->accel_x = accel_x_raw / 8192.0f * 9.81f;  // Convert to m/s^2
    data->accel_y = accel_y_raw / 8192.0f * 9.81f;
    data->accel_z = accel_z_raw / 8192.0f * 9.81f;
    
    // Parse gyroscope (±500 deg/s, sensitivity = 65.5 LSB/(deg/s))
    int16_t gyro_x_raw = (buf[8] << 8) | buf[9];
    int16_t gyro_y_raw = (buf[10] << 8) | buf[11];
    int16_t gyro_z_raw = (buf[12] << 8) | buf[13];
    
    data->gyro_x = gyro_x_raw / 65.5f;
    data->gyro_y = gyro_y_raw / 65.5f;
    data->gyro_z = gyro_z_raw / 65.5f;
    
    // Compute pitch and roll from accelerometer
    // Pitch: rotation about Y-axis (nose up/down)
    // Roll: rotation about X-axis (wing tilt)
    data->pitch = atan2f(-data->accel_x, 
                         sqrtf(data->accel_y * data->accel_y + 
                               data->accel_z * data->accel_z)) * (180.0f / M_PI);
    
    data->roll = atan2f(data->accel_y, data->accel_z) * (180.0f / M_PI);
    
    return true;
}

// ============================================================================
// MS4525DO Pitot Tube Driver (Optional)
// ============================================================================

typedef struct {
    float differential_pressure;  // Pa
    float temperature;            // °C
    float airspeed;               // m/s
} pitot_data_t;

static bool pitot_initialized = false;

/**
 * Initialize MS4525DO pitot tube sensor
 */
bool pitot_init(void) {
    // Try to read from sensor
    uint8_t buf[4];
    if (i2c_read_blocking(I2C_PORT, MS4525DO_ADDR, buf, 4, false) < 0) {
        printf("[Pitot] Sensor not found (optional)\n");
        return false;
    }
    
    pitot_initialized = true;
    printf("[Pitot] MS4525DO initialized\n");
    
    return true;
}

/**
 * Read differential pressure and compute airspeed
 */
bool pitot_read(pitot_data_t *data) {
    if (!pitot_initialized) {
        data->differential_pressure = 0.0f;
        data->temperature = 25.0f;
        data->airspeed = 0.0f;
        return false;
    }
    
    uint8_t buf[4];
    if (i2c_read_blocking(I2C_PORT, MS4525DO_ADDR, buf, 4, false) < 0) {
        return false;
    }
    
    // Parse pressure (14-bit, status in top 2 bits)
    uint8_t status = (buf[0] >> 6) & 0x03;
    if (status != 0) {
        return false;  // Stale data or fault
    }
    
    uint16_t pressure_raw = ((buf[0] & 0x3F) << 8) | buf[1];
    uint16_t temp_raw = ((buf[2] << 8) | buf[3]) >> 5;
    
    // Convert to physical units
    // MS4525DO-DS3BS002DP: ±2 psi range
    float psi = ((float)pressure_raw - 8192.0f) / 8192.0f * 2.0f;
    data->differential_pressure = psi * 6894.76f;  // Convert psi to Pa
    
    data->temperature = ((float)temp_raw * 200.0f / 2048.0f) - 50.0f;
    
    // Compute airspeed using Bernoulli: v = sqrt(2 * dP / rho)
    // Air density at sea level: 1.225 kg/m³
    float rho = 1.225f;
    if (data->differential_pressure > 0) {
        data->airspeed = sqrtf(2.0f * data->differential_pressure / rho);
    } else {
        data->airspeed = 0.0f;
    }
    
    return true;
}

// ============================================================================
// Complementary Filter for Pitch/Roll
// ============================================================================

typedef struct {
    float pitch;
    float roll;
    float alpha;  // Filter coefficient (0-1)
    absolute_time_t last_time;
} complementary_filter_t;

static complementary_filter_t cf = {
    .pitch = 0.0f,
    .roll = 0.0f,
    .alpha = 0.98f,  // 98% gyro, 2% accel
    .last_time = {0}
};

/**
 * Update complementary filter with new IMU data
 */
void complementary_filter_update(mpu6050_data_t *imu) {
    absolute_time_t now = get_absolute_time();
    
    if (cf.last_time._private_us_since_boot == 0) {
        // First reading - initialize from accelerometer
        cf.pitch = imu->pitch;
        cf.roll = imu->roll;
        cf.last_time = now;
        return;
    }
    
    // Compute dt in seconds
    float dt = absolute_time_diff_us(cf.last_time, now) / 1000000.0f;
    cf.last_time = now;
    
    if (dt > 0.1f) {
        // Too long since last update, reset
        cf.pitch = imu->pitch;
        cf.roll = imu->roll;
        return;
    }
    
    // Integrate gyroscope
    float gyro_pitch = cf.pitch + imu->gyro_y * dt;
    float gyro_roll = cf.roll + imu->gyro_x * dt;
    
    // Complementary filter
    cf.pitch = cf.alpha * gyro_pitch + (1.0f - cf.alpha) * imu->pitch;
    cf.roll = cf.alpha * gyro_roll + (1.0f - cf.alpha) * imu->roll;
}

// ============================================================================
// Main Application
// ============================================================================

int main() {
    // Initialize stdio (USB/UART)
    stdio_init_all();
    
    // Wait for USB connection (optional)
    sleep_ms(2000);
    
    printf("\n");
    printf("============================================\n");
    printf("  PINN-Lite-Foil: AoA Streaming Firmware\n");
    printf("  Raspberry Pi Pico - v1.0.0\n");
    printf("============================================\n\n");
    
    // Initialize I2C
    printf("[I2C] Initializing on GPIO %d/%d at %d Hz\n", 
           I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ_HZ);
    
    i2c_init(I2C_PORT, I2C_FREQ_HZ);
    gpio_set_function(I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_PIN);
    gpio_pull_up(I2C_SCL_PIN);
    
    // Initialize sensors
    if (!mpu6050_init()) {
        printf("[ERROR] MPU6050 initialization failed!\n");
        while (1) {
            sleep_ms(1000);
        }
    }
    
    // Optional pitot tube
    pitot_init();
    
    printf("\n[STREAM] Starting at %d Hz\n", SAMPLE_RATE_HZ);
    printf("[FORMAT] AoA,<pitch>,<roll>,<airspeed>,<timestamp_ms>\n\n");
    
    // Main sampling loop
    mpu6050_data_t imu_data;
    pitot_data_t pitot_data;
    uint32_t sample_count = 0;
    
    while (true) {
        absolute_time_t loop_start = get_absolute_time();
        
        // Read IMU
        if (mpu6050_read(&imu_data)) {
            // Update complementary filter
            complementary_filter_update(&imu_data);
            
            // Read pitot tube (if available)
            pitot_read(&pitot_data);
            
            // Get timestamp
            uint32_t timestamp_ms = to_ms_since_boot(get_absolute_time());
            
            // Output data stream
            // Format: AoA,pitch,roll,airspeed,timestamp
            printf("AoA,%.2f,%.2f,%.2f,%lu\n",
                   cf.pitch,           // Pitch (angle of attack proxy)
                   cf.roll,            // Roll
                   pitot_data.airspeed, // Airspeed (m/s)
                   timestamp_ms);
            
            sample_count++;
        }
        
        // Wait for next sample period
        absolute_time_t loop_end = get_absolute_time();
        int64_t elapsed_us = absolute_time_diff_us(loop_start, loop_end);
        int64_t sleep_us = (SAMPLE_PERIOD_MS * 1000) - elapsed_us;
        
        if (sleep_us > 0) {
            sleep_us(sleep_us);
        }
    }
    
    return 0;
}
