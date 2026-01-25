# Hardware Setup Guide

This guide covers the hardware integration for PINN-Lite-Foil edge deployment.

## Table of Contents

1. [Raspberry Pi Pico Setup](#raspberry-pi-pico-setup)
2. [MPU6050 IMU Wiring](#mpu6050-imu-wiring)
3. [Pitot Tube Integration](#pitot-tube-integration)
4. [Jetson Nano Deployment](#jetson-nano-deployment)
5. [Troubleshooting](#troubleshooting)

---

## Raspberry Pi Pico Setup

### Required Components

| Component | Description | Quantity |
|-----------|-------------|----------|
| Raspberry Pi Pico | RP2040-based microcontroller | 1 |
| MPU6050 | 6-axis IMU (accelerometer + gyroscope) | 1 |
| MS4525DO | Differential pressure sensor (pitot tube) | 1 (optional) |
| USB-C cable | For programming and data | 1 |
| Breadboard | For prototyping | 1 |
| Jumper wires | Male-to-male, various lengths | ~10 |

### Software Requirements

1. **Pico SDK** (v1.5.0+)
   ```bash
   git clone https://github.com/raspberrypi/pico-sdk.git
   cd pico-sdk
   git submodule update --init
   export PICO_SDK_PATH=$(pwd)
   ```

2. **ARM GCC Toolchain**
   ```bash
   # macOS
   brew install arm-none-eabi-gcc
   
   # Ubuntu/Debian
   sudo apt install gcc-arm-none-eabi
   ```

3. **CMake** (v3.13+)
   ```bash
   brew install cmake  # macOS
   sudo apt install cmake  # Ubuntu
   ```

### Building Firmware

```bash
cd src/deployment/pico_firmware
mkdir build && cd build
cmake .. -DPICO_SDK_PATH=$PICO_SDK_PATH
make -j$(nproc)
```

This produces `aoa_stream.uf2` in the build directory.

### Flashing the Pico

1. Hold the **BOOTSEL** button on the Pico
2. Connect USB cable to computer
3. Release BOOTSEL after ~1 second
4. Pico mounts as USB drive `RPI-RP2`
5. Copy the UF2 file:
   ```bash
   cp aoa_stream.uf2 /Volumes/RPI-RP2/  # macOS
   cp aoa_stream.uf2 /media/$USER/RPI-RP2/  # Linux
   ```
6. Pico reboots automatically

### Verifying Operation

```bash
# Find the serial port
ls /dev/cu.usb*  # macOS
ls /dev/ttyACM*  # Linux

# Connect with screen
screen /dev/cu.usbmodem* 115200
# or
screen /dev/ttyACM0 115200

# Expected output:
# AoA,5.23,2.15,12.5,1234567
# AoA,5.31,2.18,12.4,1234617
# ...
```

---

## MPU6050 IMU Wiring

### Pinout Diagram

```
Raspberry Pi Pico          MPU6050
┌──────────────┐          ┌──────────┐
│              │          │          │
│  3V3 (pin 36)├──────────┤ VCC      │
│              │          │          │
│  GND (pin 38)├──────────┤ GND      │
│              │          │          │
│  GP4 (pin 6) ├──────────┤ SDA      │
│              │          │          │
│  GP5 (pin 7) ├──────────┤ SCL      │
│              │          │          │
│              │    ┌─────┤ AD0      │ → GND for addr 0x68
│              │    │     │          │ → VCC for addr 0x69
└──────────────┘    │     └──────────┘
                    └───── GND
```

### I2C Pull-up Resistors

The Pico firmware enables internal pull-ups, but for reliable operation at 400kHz:

- Add 4.7kΩ resistors from SDA to 3.3V
- Add 4.7kΩ resistors from SCL to 3.3V

### Mounting Orientation

For correct pitch/roll calculation:

```
          +X (Forward)
            ↑
            │
   +Y ←─────┼─────→ -Y
  (Left)    │     (Right)
            │
            ↓
          -X (Backward)

+Z points UP (away from board)
```

**Mount with:**
- X-axis pointing towards nose of aircraft
- Y-axis pointing towards left wing
- Z-axis pointing up

---

## Pitot Tube Integration

### MS4525DO Differential Pressure Sensor

The MS4525DO is an I2C differential pressure sensor suitable for airspeed measurement.

### Recommended Model

**MS4525DO-DS3BS002DP**
- Range: ±2 psi (±13.8 kPa)
- Resolution: 14-bit
- I2C address: 0x28

### Wiring

```
Raspberry Pi Pico          MS4525DO
┌──────────────┐          ┌──────────┐
│              │          │          │
│  3V3 (pin 36)├──────────┤ VDD      │
│              │          │          │
│  GND (pin 38)├──────────┤ GND      │
│              │          │          │
│  GP4 (pin 6) ├──────────┤ SDA      │ ← Same I2C bus
│              │          │          │
│  GP5 (pin 7) ├──────────┤ SCL      │ ← as MPU6050
│              │          │          │
└──────────────┘          └──────────┘
```

### Pitot Tube Assembly

```
                  Static port ─────────┐
                                       ↓
    ┌────────────────────────┐    ┌────────┐
    │  Dynamic (ram) port    │    │ Static │
    │          ○             │    │   ○    │
    └──────────│─────────────┘    └────│───┘
               │                       │
               │    ┌──────────────────┤
               │    │                  │
               ▼    ▼                  ▼
           ┌───────────┐           ┌──────┐
           │ MS4525DO  │           │ Open │
           │  P1 (+)   │           │ (atm)│
           └───────────┘           └──────┘
               Port A               Port B
         (dynamic pressure)    (static pressure)
```

### Calibration Procedure

1. **Zero Offset Calibration**
   - Block both ports (no airflow)
   - Record 100 samples
   - Average = zero offset

2. **Span Calibration** (optional)
   - Use reference manometer
   - Apply known pressure differential
   - Calculate scale factor

### Airspeed Calculation

```c
// Bernoulli equation
// V = sqrt(2 * dP / rho)
// where:
//   dP = differential pressure (Pa)
//   rho = air density (kg/m³) ≈ 1.225 at sea level

float airspeed_mps = sqrtf(2.0f * diff_pressure_pa / 1.225f);
```

---

## Jetson Nano Deployment

### Requirements

- NVIDIA Jetson Nano Developer Kit (4GB recommended)
- microSD card (64GB+, Class 10)
- 5V 4A power supply (barrel jack)
- USB WiFi adapter or Ethernet

### JetPack Setup

1. Download JetPack 4.6+ from NVIDIA
2. Flash to microSD using balenaEtcher
3. Boot and complete initial setup

### Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python packages
pip3 install --upgrade pip
pip3 install numpy onnxruntime

# For GPU acceleration (optional)
pip3 install onnxruntime-gpu
```

### Deploy PINN Model

```bash
# Copy ONNX model to Jetson
scp models/onnx/student_compressed.onnx jetson@<IP>:~/pinn/

# Test inference
python3 -c "
import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession('~/pinn/student_compressed.onnx')
input_name = sess.get_inputs()[0].name

# Benchmark
times = []
for _ in range(1000):
    start = time.perf_counter()
    sess.run(None, {input_name: np.random.randn(1, 3).astype(np.float32)})
    times.append((time.perf_counter() - start) * 1000)

print(f'Mean latency: {np.mean(times):.3f} ms')
print(f'P99 latency: {np.percentile(times, 99):.3f} ms')
"
```

### Serial Communication with Pico

```python
import serial
import time

# Connect to Pico
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

while True:
    line = ser.readline().decode('utf-8').strip()
    if line.startswith('AoA,'):
        parts = line.split(',')
        pitch = float(parts[1])
        roll = float(parts[2])
        airspeed = float(parts[3])
        
        print(f"Pitch: {pitch:.2f}°, Roll: {roll:.2f}°, Airspeed: {airspeed:.1f} m/s")
```

---

## Troubleshooting

### Common Issues

#### MPU6050 Not Detected

```bash
# Check I2C devices
i2cdetect -y 1

# Expected output:
#      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
# 60: -- -- -- -- -- -- -- -- 68 -- -- -- -- -- -- --
```

**Solutions:**
- Check wiring (SDA/SCL not swapped)
- Verify 3.3V power
- Add external pull-up resistors
- Check AD0 pin connection

#### Noisy IMU Readings

**Solutions:**
- Increase DLPF bandwidth (reduce noise, increase latency)
- Add software filtering (complementary filter, Kalman)
- Isolate sensor from vibration

#### Pitot Readings Incorrect

**Solutions:**
- Check for tube blockage
- Ensure proper port connection (dynamic vs static)
- Perform zero-offset calibration
- Verify I2C address (0x28)

#### Pico Not Mounting as USB Drive

**Solutions:**
- Use different USB cable (data-capable)
- Hold BOOTSEL before connecting
- Try different USB port
- Check if Pico is already in application mode

### LED Status Codes

| Pattern | Meaning |
|---------|---------|
| Solid ON | Normal operation |
| Slow blink | I2C error |
| Fast blink | Sensor initialization failed |
| Off | No power or crashed |

---

## Safety Considerations

⚠️ **WARNING: This system is for research/educational purposes only.**

- Not certified for flight-critical applications
- Always have backup instrumentation
- Test thoroughly before any flight operations
- Follow local regulations for unmanned aircraft

---

## Next Steps

1. [Calibration Guide](calibration.md)
2. [Integration with Flight Controller](flight_controller.md)
3. [Real-Time Data Logging](data_logging.md)
