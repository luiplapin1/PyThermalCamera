# PyThermalcam v5.1
Python Software to use the Topdon TC001 Thermal Camera on Linux and the Raspberry Pi. It **may** work with other similar cameras! Please feed back if it does!

Huge kudos to LeoDJ on the EEVBlog forum for reverse engineering the image format from these kind of cameras (InfiRay P2 Pro) to get the raw temperature data!
https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
Check out Leo's Github here: https://github.com/LeoDJ/P2Pro-Viewer/tree/main

## Introduction

This is a comprehensive Python implementation of Thermal Camera software for the Topdon TC001!
(https://www.amazon.co.uk/dp/B0BBRBMZ58)

**NEW in v5.1:** Enhanced functionality with measurement points, automatic data logging, improved camera detection, and image rotation correction.

No commands are sent to the camera - instead, we take the raw video feed, do some OpenCV processing, and display a rich heatmap interface with temperature analysis tools.

![Screenshot](media/TC00120230701-131032.png)

This program, and associated information is Open Source (see Licence), but if you have gotten value from these kinds of projects and think they are worth something, please consider donating: https://paypal.me/leslaboratory?locale.x=en_GB 

This readme is accompanied by youtube videos. Visit my Youtube Channel at: https://www.youtube.com/leslaboratory

The video is here: https://youtu.be/PiVwZoQ8_jQ

## What's New in v5.1

- **Interactive Measurement Points**: Click to add up to 8 temperature measurement points anywhere on the image
- **Automatic Data Logging**: Continuous CSV logging of temperature data with configurable intervals
- **Frame Capture System**: Automatic saving of thermal frames with timestamp correlation
- **Enhanced Camera Detection**: Multiple access methods tested automatically for better compatibility
- **Image Rotation Correction**: Proper 180° rotation to correct camera orientation
- **Improved Error Handling**: Better camera access testing and error recovery
- **Measurement Point HUD**: Dedicated display area showing all measurement point temperatures
- **Snapshot with Measurements**: Save images with measurement point data to text files

## Features

Tested on Debian - all features working correctly. Also tested on Raspberry Pi with workarounds for OpenCV compilation issues.

### Core Thermal Processing
- **Native Resolution**: 256×192 thermal sensor data processing
- **Temperature Calculations**: Accurate conversion from raw thermal data to Celsius
- **Real-time Analysis**: Center point, min/max, and average temperature monitoring
- **Bicubic Interpolation**: Scale factor 1-5x with optional blur smoothing

### Interactive Features
- **Measurement Points** (NEW): Left-click to add, right-click to remove temperature measurement points
- **Live Temperature Display**: Real-time temperature readings for all measurement points
- **Crosshair Center Point**: Always-visible center temperature reference
- **Floating Min/Max**: Threshold-based extreme temperature highlighting

### Display and Recording
- **Multiple Colormaps**: 11 different false-color schemes (Jet, Hot, Magma, Inferno, Plasma, Bone, Spring, Autumn, Viridis, Parula, Rainbow)
- **Adjustable Contrast**: Variable contrast control (0.0-3.0)
- **Fullscreen/Windowed**: Toggle between display modes
- **Video Recording**: Save thermal sessions as AVI files
- **Snapshot Capture**: PNG image saves with measurement data export

### Data Logging (NEW)
- **Automatic CSV Logging**: Continuous temperature data recording
- **Configurable Intervals**: Set capture frequency (default: 3 seconds)
- **Frame Correlation**: Each CSV entry linked to saved thermal image
- **Measurement Export**: Snapshot measurement points saved to text files

### User Interface
- **Main HUD**: Displays avg temp, settings, and status information
- **Measurement HUD**: Dedicated panel for measurement point data
- **Keyboard Controls**: Full keyboard interface for all functions
- **Mouse Interaction**: Point-and-click measurement system

## Dependencies

**Required:**
- Python 3
- OpenCV for Python

**Installation:**
```bash
sudo apt-get install python3-opencv
```

**Optional:** v4l-utils for device detection
```bash
sudo apt-get install v4l-utils
```

## Running the Program

### Available Scripts

**tc001-RAW.py** - Basic thermal camera frame grabber (starting point for custom development)

**tc001v5.1.py** - Full-featured thermal analysis application (main program)

### Basic Usage

1. Connect your thermal camera
2. Find device number:
   ```bash
   v4l2-ctl --list-devices
   ```
3. Run the program:
   ```bash
   python3 tc001v5.1.py --device 0
   ```

### Command Line Options

```bash
python3 tc001v5.1.py --device 0 --interval 1.0 --test-only
```

- `--device N`: Video device number (default: 0)
- `--interval X`: Data capture interval in seconds (default: 3.0)
- `--test-only`: Test camera access without running full interface

### Camera Detection

v5.1 includes automatic camera detection that tests multiple access methods:
- Standard OpenCV capture
- V4L2 device path access
- V4L2 backend specification
- GStreamer backend (if available)

If the program cannot access your camera, it will display diagnostic information to help troubleshoot connectivity issues.

## Controls

### Keyboard Controls

**Image Processing:**
- `a` / `z`: Increase/Decrease blur radius
- `d` / `c`: Change scale multiplier (1-5x) - *Note: Window resize may not work on Pi*
- `f` / `v`: Increase/Decrease contrast

**Display:**
- `q` / `w`: Toggle fullscreen/windowed mode - *Note: Return to windowed may not work on Pi*
- `h`: Toggle main HUD display
- `m`: Cycle through color maps

**Temperature Analysis:**
- `s` / `x`: Adjust floating temperature label threshold
- `n`: Clear all measurement points

**Recording & Capture:**
- `r` / `t`: Start/Stop video recording (saves as AVI)
- `p`: Take snapshot (saves PNG + measurement data)

**System:**
- `ESC`: Exit program

### Mouse Controls (NEW in v5.1)

- **Left Click**: Add measurement point (up to 8 points)
- **Right Click**: Remove closest measurement point

## Data Output

### Automatic CSV Logging
- **File**: `temperature_data.csv` (created automatically)
- **Columns**: Timestamp, Max Temp, Max Coordinates, Min Temp, Min Coordinates, Avg Temp, Frame File
- **Updates**: Every configured interval (default: 3 seconds)

### Frame Capture
- **Directory**: `captured_frames/` (auto-created)
- **Format**: PNG files with timestamp correlation to CSV data
- **Naming**: `frame_TIMESTAMP.png`

### Manual Snapshots
- **Directory**: `snapshots/` (auto-created)  
- **Thermal Image**: `TC001YYYYMMDD-HHMMSS.png`
- **Measurement Data**: `TC001YYYYMMDD-HHMMSS_measurements.txt` (if measurement points exist)

## HUD Information Display

### Main HUD (Top Left)
- Average scene temperature
- Label threshold setting
- Current colormap
- Blur radius
- Scaling multiplier  
- Contrast value
- Last snapshot time
- Recording status with elapsed time

### Measurement HUD (Top Right) - NEW
- Up to 8 measurement points with live temperatures
- Point coordinates in thermal image space
- Color-coded point identification

## Technical Notes

### Raspberry Pi Compatibility
- All features functional with workarounds
- OpenCV window resizing limitations noted
- Fullscreen return may require restart
- Performance optimized for Pi hardware

### Camera Data Processing
- **Color Conversion**: YUV to BGR with thermal data preservation
- **Temperature Calculation**: Raw sensor values converted using `(raw/64) - 273.15`
- **Image Correction**: 180° rotation applied to both thermal and visual data
- **Data Splitting**: Automatic separation of visual and thermal data from camera stream

### Error Handling
- Comprehensive camera access testing
- Frame processing error recovery
- Automatic fallback methods for different camera configurations
- Detailed diagnostic output for troubleshooting

## Troubleshooting

### Camera Access Issues
1. Run with `--test-only` flag to diagnose connectivity
2. Check device permissions: `ls -l /dev/video*`
3. Try different device numbers: `--device 0`, `--device 1`, etc.
4. Verify camera connection and power

### Performance Issues
- Reduce scale factor for better frame rates
- Disable blur for faster processing
- Lower capture interval for less frequent data logging
- Consider threading improvements for multi-core systems

## TODO / Future Enhancements

- **Code Optimization**: Refactoring and performance improvements
- **Threading**: Multi-threaded processing for better Pi performance  
- **Temperature Graphing**: Historical temperature trend visualization
- **Advanced Analysis**: Statistical analysis tools for measurement data
- **Export Options**: Additional data export formats
- **UI Improvements**: Enhanced measurement point management
- **Calibration Tools**: Temperature calibration and offset controls

## Version History

- **v4.2**: Original stable release with basic thermal processing
- **v5.1**: Major update with interactive measurement points, data logging, improved camera handling, and enhanced user interface

---

*For support and updates, visit: https://youtube.com/leslaboratory*
