# PINN-Lite-Foil Calibration Guide

## Overview

This guide covers sensor calibration procedures for the PINN-Lite-Foil system, ensuring accurate angle-of-attack and airspeed measurements for real-time flow prediction.

---

## 1. MPU6050 IMU Calibration

### 1.1 Accelerometer Offset Calibration

The accelerometer must be calibrated to remove manufacturing offsets.

**Procedure:**

1. Place the sensor on a **perfectly level surface**
2. Collect 1000 samples at rest
3. Average the readings for each axis

```c
// Calibration constants (example values)
#define ACCEL_X_OFFSET  (-0.023f)  // g
#define ACCEL_Y_OFFSET  (0.015f)   // g  
#define ACCEL_Z_OFFSET  (-0.042f)  // g (should read ~1.0g when level)

// Apply calibration
float ax_calibrated = ax_raw - ACCEL_X_OFFSET;
float ay_calibrated = ay_raw - ACCEL_Y_OFFSET;
float az_calibrated = az_raw - ACCEL_Z_OFFSET;
```

**Expected Results:**
- X, Y axes: Should read 0g ± 0.01g when level
- Z axis: Should read 1.0g ± 0.01g when level

### 1.2 Gyroscope Offset Calibration

Gyroscopes exhibit bias drift that must be removed.

**Procedure:**

1. Keep the sensor **completely stationary**
2. Collect 2000 samples (at 100Hz = 20 seconds)
3. Average the readings for each axis

```c
// Calibration constants (example values in deg/s)
#define GYRO_X_OFFSET  (1.234f)
#define GYRO_Y_OFFSET  (-0.567f)
#define GYRO_Z_OFFSET  (0.891f)

// Apply calibration
float gx_calibrated = gx_raw - GYRO_X_OFFSET;
float gy_calibrated = gy_raw - GYRO_Y_OFFSET;
float gz_calibrated = gz_raw - GYRO_Z_OFFSET;
```

**Expected Results:**
- All axes: Should read 0 deg/s ± 0.1 deg/s when stationary

### 1.3 Temperature Compensation

IMU offsets drift with temperature. For precision applications:

```c
// Linear temperature compensation model
float temp_correction(float offset_20c, float temp_coeff, float current_temp) {
    float delta_temp = current_temp - 20.0f;  // Reference at 20°C
    return offset_20c + (temp_coeff * delta_temp);
}

// Example coefficients (device-specific)
#define GYRO_X_TEMP_COEFF  (0.01f)  // deg/s per °C
```

**Calibration Procedure:**
1. Perform offset calibration at 10°C, 20°C, 30°C
2. Fit linear regression to find temperature coefficient
3. Store coefficients in firmware

---

## 2. MS4525DO Pitot Tube Calibration

### 2.1 Zero-Point Offset

The differential pressure sensor requires zero-point calibration at rest.

**Procedure:**

1. Ensure **no airflow** over the pitot tube
2. Cover both pressure ports with caps (optional but recommended)
3. Collect 500 samples
4. Average to find zero offset

```c
// Raw ADC reading at zero differential pressure
#define PITOT_ZERO_OFFSET  2048  // 12-bit ADC midpoint typical

// Convert to differential pressure (Pa)
float pressure_pa = ((raw_adc - PITOT_ZERO_OFFSET) / 4096.0f) * PRESSURE_RANGE_PA;
```

### 2.2 Span Calibration

For accurate airspeed, validate against a known reference.

**Equipment Needed:**
- Calibrated manometer (±0.1 Pa accuracy)
- Pressure source (hand pump or wind tunnel)

**Procedure:**

1. Apply known differential pressures: 0, 50, 100, 200, 500 Pa
2. Record sensor readings at each point
3. Calculate scale factor

```c
// Span correction
#define PITOT_SCALE_FACTOR  1.023f  // Adjust based on calibration

float pressure_calibrated = (raw_pressure - PITOT_ZERO_OFFSET) * PITOT_SCALE_FACTOR;
```

### 2.3 Airspeed Calculation

Convert differential pressure to airspeed using Bernoulli's equation:

$$V = \sqrt{\frac{2 \Delta P}{\rho}}$$

Where:
- $V$ = airspeed (m/s)
- $\Delta P$ = differential pressure (Pa)
- $\rho$ = air density (kg/m³)

```c
// Standard air density at sea level, 15°C
#define AIR_DENSITY_STD  1.225f  // kg/m³

// Correct for altitude and temperature
float air_density_corrected(float pressure_alt_m, float temp_c) {
    // Barometric formula approximation
    float temp_k = temp_c + 273.15f;
    float pressure_ratio = powf(1.0f - 0.0065f * pressure_alt_m / 288.15f, 5.2561f);
    return AIR_DENSITY_STD * pressure_ratio * (288.15f / temp_k);
}

float calc_airspeed(float diff_pressure_pa, float air_density) {
    if (diff_pressure_pa < 0) return 0;
    return sqrtf(2.0f * diff_pressure_pa / air_density);
}
```

---

## 3. Angle-of-Attack Calibration

### 3.1 Mounting Alignment

The IMU must be aligned with the aircraft body frame.

**Installation Check:**
1. Mount aircraft on leveling fixture
2. Set wings perfectly level
3. Record IMU pitch reading → **mounting offset**

```c
#define PITCH_MOUNTING_OFFSET  (2.5f)  // degrees nose-up bias

float aoa_corrected = pitch_measured - PITCH_MOUNTING_OFFSET;
```

### 3.2 Wind Tunnel Validation

For flight-critical applications, validate AoA against wind tunnel measurements.

**Procedure:**

1. Mount aircraft model in wind tunnel with force balance
2. Sweep AoA from -5° to +20° in 1° increments
3. Record:
   - IMU-measured pitch angle
   - Force balance AoA reference
   - Lift coefficient
4. Create correction lookup table

```python
# Python calibration script
import numpy as np

# Measured data
imu_aoa = np.array([-4.8, -3.9, ..., 19.2])  # IMU readings
ref_aoa = np.array([-5.0, -4.0, ..., 20.0])  # Reference

# Fit polynomial correction
coeffs = np.polyfit(imu_aoa, ref_aoa, deg=2)
# e.g., coeffs = [0.001, 1.02, 0.15]

# Apply correction
def correct_aoa(imu_reading):
    return coeffs[0]*imu_reading**2 + coeffs[1]*imu_reading + coeffs[2]
```

---

## 4. System Integration Calibration

### 4.1 Timing Synchronization

Ensure sensor data is properly timestamped for PINN inference.

**Check Procedure:**

1. Apply known impulse (sharp rotation)
2. Verify IMU and airspeed timestamps align within ±5ms
3. Adjust delays if needed

```c
// Add timestamp to each measurement
typedef struct {
    float pitch;
    float roll;
    float airspeed;
    uint32_t timestamp_ms;
} sensor_packet_t;
```

### 4.2 End-to-End Validation

Validate the complete pipeline before deployment.

**Test Cases:**

| Condition | Expected AoA | Expected Airspeed |
|-----------|--------------|-------------------|
| Level, 20 m/s | 0° ± 1° | 20 ± 0.5 m/s |
| 10° climb, 15 m/s | 10° ± 1° | 15 ± 0.5 m/s |
| Hover (if VTOL) | N/A | 0 ± 0.2 m/s |

---

## 5. Calibration Data Storage

### 5.1 EEPROM Format

Store calibration in non-volatile memory:

```c
typedef struct {
    uint32_t magic;           // 0xCAFE1234
    uint16_t version;         // Calibration format version
    
    // IMU calibration
    float accel_offset[3];    // X, Y, Z
    float gyro_offset[3];     // X, Y, Z
    float gyro_temp_coeff[3]; // Temperature compensation
    
    // Pitot calibration
    int16_t pitot_zero_offset;
    float pitot_scale_factor;
    
    // Mounting
    float pitch_mounting_offset;
    float roll_mounting_offset;
    
    uint32_t crc32;           // Data integrity check
} calibration_data_t;
```

### 5.2 Loading at Boot

```c
bool load_calibration(calibration_data_t* cal) {
    // Read from flash/EEPROM
    flash_read(CAL_FLASH_ADDR, cal, sizeof(*cal));
    
    // Verify magic number
    if (cal->magic != 0xCAFE1234) {
        printf("WARNING: No calibration found, using defaults\n");
        return false;
    }
    
    // Verify CRC
    uint32_t computed_crc = crc32(cal, sizeof(*cal) - 4);
    if (computed_crc != cal->crc32) {
        printf("ERROR: Calibration data corrupted!\n");
        return false;
    }
    
    return true;
}
```

---

## 6. Troubleshooting

### Issue: AoA drifts over time

**Causes:**
- Gyroscope bias drift
- Temperature change

**Solutions:**
1. Increase complementary filter accelerometer weight
2. Implement temperature compensation
3. Add periodic in-flight recalibration (during level flight)

### Issue: Airspeed reads non-zero at rest

**Causes:**
- Pitot zero offset incorrect
- Water/debris in pressure lines

**Solutions:**
1. Recalibrate zero offset
2. Purge pressure lines with dry air
3. Check for loose fittings

### Issue: Noisy measurements

**Causes:**
- Electrical interference
- Vibration
- Poor grounding

**Solutions:**
1. Use shielded cables
2. Add vibration damping mounts
3. Ensure proper ground connections
4. Increase filter time constants

---

## Quick Reference

| Parameter | Typical Value | Acceptable Range |
|-----------|---------------|------------------|
| Accel offset | ±0.05g | ±0.1g |
| Gyro offset | ±2°/s | ±5°/s |
| Pitot zero | 2048 (12-bit) | ±100 counts |
| Mounting offset | ±5° | ±15° |
| AoA accuracy | ±0.5° | ±1.0° |
| Airspeed accuracy | ±0.3 m/s | ±0.5 m/s |
