#!/usr/bin/env python3
'''
Les Wright 21 June 2023 - Modified to solve OpenCV problems
https://youtube.com/leslaboratory
A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!
MODIFIED: Added functionality to mark temperature measurement points
'''

import cv2
import numpy as np
import argparse
import time
import io
import sys
import csv  # Import module to handle CSV
import os  # Add for directory management

# Global variables for measurement points
measurement_points = []
max_measurement_points = 8
current_thdata = None
current_scale = 3

# File paths
csv_file = "temperature_data.csv"
frames_dir = "captured_frames"

def initialize_csv():
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        with open(csv_file, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Max Temp", "Max Coords", "Min Temp", "Min Coords", "Avg Temp", "Frame File"])
    except FileExistsError:
        try:
            with open(csv_file, 'r') as file:
                first_line = file.readline().strip()
                if not first_line.startswith("Timestamp"):
                    with open(csv_file, 'r+') as f:
                        content = f.read()
                        f.seek(0, 0)
                        f.write("Timestamp,Max Temp,Max Coords,Min Temp,Min Coords,Avg Temp,Frame File\n" + content)
        except:
            pass

def save_to_csv(timestamp, max_temp, max_coords, min_temp, min_coords, avg_temp, frame_file):
    """Save data to CSV file."""
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Convert numpy coordinates to simple tuples without quotes
        max_coords_clean = f"({int(max_coords[0])},{int(max_coords[1])})"
        min_coords_clean = f"({int(min_coords[0])},{int(min_coords[1])})"
        writer.writerow([timestamp, max_temp, max_coords_clean, min_temp, min_coords_clean, avg_temp, frame_file])

def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): 
                return True
    except Exception: 
        pass
    return False

def mouse_callback(event, x, y, flags, param):
    """Callback to handle mouse clicks and add measurement points"""
    global measurement_points, max_measurement_points, current_scale
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert screen coordinates to thermal image coordinates
        thermal_x = int(x / current_scale)
        thermal_y = int(y / current_scale)
        
        # Check if within bounds
        if 0 <= thermal_x < 256 and 0 <= thermal_y < 192:
            if len(measurement_points) < max_measurement_points:
                measurement_points.append((thermal_x, thermal_y, x, y))
                print(f"Measurement point added: ({thermal_x}, {thermal_y})")
            else:
                print(f"Maximum of {max_measurement_points} points reached. Press 'n' to clear.")
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click: remove closest point
        if measurement_points:
            distances = [((x - px)**2 + (y - py)**2)**0.5 for _, _, px, py in measurement_points]
            closest_idx = distances.index(min(distances))
            removed_point = measurement_points.pop(closest_idx)
            print(f"Measurement point removed: ({removed_point[0]}, {removed_point[1]})")

def get_temperature_at_point(thdata, x, y):
    """Get temperature at a specific point"""
    try:
        if thdata is None or len(thdata.shape) < 2:
            return None
        
        # Check bounds
        if y >= thdata.shape[0] or x >= thdata.shape[1]:
            return None
            
        if len(thdata.shape) > 2 and thdata.shape[2] > 1:
            hi = int(np.clip(thdata[y][x][0], 0, 255))
            lo = int(np.clip(thdata[y][x][1], 0, 255))
        else:
            hi = int(np.clip(thdata[y][x], 0, 255))
            lo = 0
        
        rawtemp = hi + (lo * 256)
        temp = (rawtemp / 64) - 273.15
        return round(temp, 2)
    except Exception as e:
        print(f"Error calculating temperature at point ({x}, {y}): {e}")
        return None

def draw_measurement_points(image, thdata, scale):
    """Draw measurement points and their temperatures"""
    global measurement_points
    
    colors = [
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 128, 255),    # Orange
        (255, 255, 255),  # White
        (128, 0, 128)     # Purple
    ]
    
    for i, (thermal_x, thermal_y, screen_x, screen_y) in enumerate(measurement_points):
        color = colors[i % len(colors)]
        
        # Draw circle
        cv2.circle(image, (screen_x, screen_y), 6, (0, 0, 0), 2)
        cv2.circle(image, (screen_x, screen_y), 6, color, -1)
        cv2.circle(image, (screen_x, screen_y), 8, (255, 255, 255), 1)
        
        # Get and display temperature
        temp = get_temperature_at_point(thdata, thermal_x, thermal_y)
        if temp is not None:
            # Point number
            cv2.putText(image, str(i + 1), (screen_x - 4, screen_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(i + 1), (screen_x - 4, screen_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Temperature without degree symbol
            temp_text = f'{temp}C'
            text_x = screen_x + 12
            text_y = screen_y - 8
            
            # Adjust position if going off screen
            if text_x + 60 > image.shape[1]:
                text_x = screen_x - 70
            if text_y < 15:
                text_y = screen_y + 20
            
            cv2.putText(image, temp_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, temp_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def draw_measurement_hud(image, thdata):
    """Draw HUD with measurement points information"""
    global measurement_points
    
    if not measurement_points:
        return
    
    # Background for measurements HUD
    hud_height = min(len(measurement_points) * 16 + 20, 150)
    cv2.rectangle(image, (170, 0), (400, hud_height), (0, 0, 0), -1)
    
    cv2.putText(image, 'Measurement Points:', (180, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    
    colors = [
        (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0),
        (255, 0, 0), (0, 128, 255), (255, 255, 255), (128, 0, 128)
    ]
    
    for i, (thermal_x, thermal_y, _, _) in enumerate(measurement_points):
        if i >= 8:  # Maximum 8 lines in HUD
            break
            
        temp = get_temperature_at_point(thdata, thermal_x, thermal_y)
        if temp is not None:
            color = colors[i % len(colors)]
            text = f'P{i+1}: {temp}C ({thermal_x},{thermal_y})'
            cv2.putText(image, text, (180, 30 + i * 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

def update_measurement_points_scale(new_scale):
    """Update screen coordinates of points when changing scale"""
    global measurement_points, current_scale
    
    if current_scale == new_scale:
        return
    
    scale_ratio = new_scale / current_scale
    updated_points = []
    
    for thermal_x, thermal_y, screen_x, screen_y in measurement_points:
        new_screen_x = int(thermal_x * new_scale)
        new_screen_y = int(thermal_y * new_scale)
        updated_points.append((thermal_x, thermal_y, new_screen_x, new_screen_y))
    
    measurement_points = updated_points
    current_scale = new_scale

def test_camera_access(device_num):
    """Test different methods to access the camera"""
    methods = [
        ("Default", lambda d: cv2.VideoCapture(d)),
        ("V4L2 Path", lambda d: cv2.VideoCapture(f'/dev/video{d}')),
        ("V4L2 Backend", lambda d: cv2.VideoCapture(f'/dev/video{d}', cv2.CAP_V4L2)),
        ("GStreamer", lambda d: cv2.VideoCapture(f'/dev/video{d}', cv2.CAP_GSTREAMER)),
    ]
    
    for name, method in methods:
        try:
            cap = method(device_num)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ {name} works - Frame shape: {frame.shape}")
                    cap.release()
                    return method(device_num)
                else:
                    print(f"✗ {name} opens but doesn't capture frames")
            else:
                print(f"✗ {name} cannot open device")
            cap.release()
        except Exception as e:
            print(f"✗ {name} error: {e}")
    
    return None

def main():
    global current_thdata, current_scale
    
    print('TC001 Thermal Camera - Script with Measurement Points')
    print('Testing different camera access methods...')
    print('')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Video Device number")
    parser.add_argument("--test-only", action="store_true", help="Only test access without running full program")
    parser.add_argument("--interval", type=float, default=3.0, help="Capture interval in seconds (default: 1.0)")
    args = parser.parse_args()
    
    dev = args.device
    capture_interval = args.interval
    isPi = is_raspberrypi()
    
    print(f"Testing device: /dev/video{dev}")
    print(f"Raspberry Pi detected: {isPi}")
    print(f"Capture interval: {capture_interval} seconds")
    print("-" * 50)
    
    # Test camera access
    cap = test_camera_access(dev)
    
    if cap is None:
        print("\n❌ Could not access camera with any method.")
        print("Suggestions:")
        print("- Verify camera is connected")
        print("- Try different device numbers: --device 0, --device 1, etc.")
        print("- Run 'v4l2-ctl --list-devices' to see available devices")
        print("- Check permissions: 'ls -l /dev/video*'")
        return
    
    print(f"✅ Camera accessible!")
    
    if args.test_only:
        cap.release()
        return
    
    # Camera configuration
    print("Configuring camera...")
    
    # DO NOT automatically convert to RGB to preserve thermal data
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
    
    # Try to set expected resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)  # 192*2 for image + thermal data
    
    # Check current configuration
    width_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_actual = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Configured resolution: {width_actual}x{height_actual}")
    
    # Application parameters
    width = 256
    height = 192
    scale = 3
    current_scale = scale
    newWidth = width * scale
    newHeight = height * scale
    alpha = 1.0
    colormap = 0
    dispFullscreen = False
    rad = 0
    threshold = 2
    hud = True
    recording = False
    elapsed = "00:00:00"
    snaptime = "None"
    
    cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('Thermal', newWidth, newHeight)
    
    # Set mouse callback
    cv2.setMouseCallback('Thermal', mouse_callback)
    
    print("Controls:")
    print("a/z: Increase/Decrease Blur")
    print("s/x: High/Low temperature threshold")  
    print("d/c: Change scale")
    print("f/v: Contrast")
    print("q/w: Fullscreen/Window")
    print("r/t: Record/Stop")
    print("p: Capture")
    print("m: Change colormap")
    print("h: Toggle HUD")
    print("n: Clear measurement points")
    print("MOUSE:")
    print("  Left click: Add measurement point")
    print("  Right click: Remove closest point")
    print("ESC: Exit")
    print(f"AUTOMATIC CAPTURE: Every {capture_interval} seconds")
    print("-" * 50)
    
    frame_count = 0
    error_count = 0

    initialize_csv()  # Initialize CSV file at program start
    last_save_time = time.time()  # Time of last saved capture
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            error_count += 1
            print(f"Error reading frame #{frame_count}, errors: {error_count}")
            if error_count > 10:
                print("Too many errors, exiting...")
                break
            time.sleep(0.1)
            continue
        
        if frame is None:
            print("Frame is None")
            continue
            
        frame_count += 1
        error_count = 0  # Reset error count on successful read
        
        # Debug: show frame info only for first frames
        if frame_count <= 3:
            print(f"Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}")
            if len(frame.shape) == 3:
                print(f"  Min values per channel: {frame.min(axis=(0,1))}")
                print(f"  Max values per channel: {frame.max(axis=(0,1))}")
        
        try:
            # Check frame dimensions
            if len(frame.shape) != 3:
                print(f"Unexpected frame shape: {frame.shape}")
                continue
                
            frame_height, frame_width = frame.shape[:2]
            
            # 1) Split frame into image and thermogram
            if frame_height >= 384:  # Full frame with thermal data
                mid_point = frame_height // 2
                imdata = frame[:mid_point]
                thdata = frame[mid_point:]
                current_thdata = thdata  # Save for measurement points
            else:
                print(f"Frame height {frame_height} is less than expected (384)")
                imdata = frame
                thdata = frame  # Use same frame as fallback
                current_thdata = thdata
            
            # 2) NOW rotate each block 180°
            imdata = cv2.rotate(imdata, cv2.ROTATE_180)
            thdata = cv2.rotate(thdata, cv2.ROTATE_180)
            current_thdata = thdata
            
            # 3) Process thermal data if available
            if thdata.shape[0] >= 96 and thdata.shape[1] >= 128:
                # Central pixel temperature
                try:
                    # Ensure values are in correct range
                    hi = int(thdata[96][128][0]) if len(thdata.shape) > 2 else int(thdata[96][128])
                    lo = int(thdata[96][128][1]) if len(thdata.shape) > 2 and thdata.shape[2] > 1 else 0
                    
                    # Clamp values to avoid overflow
                    hi = np.clip(hi, 0, 255)
                    lo = np.clip(lo, 0, 255)
                    
                    rawtemp = hi + (lo * 256)
                    temp = (rawtemp / 64) - 273.15
                    temp = round(temp, 2)
                except (IndexError, TypeError, OverflowError) as e:
                    print(f"Error calculating central temperature: {e}")
                    temp = 20.0  # Default value
                
                # Maximum temperature
                try:
                    if len(thdata.shape) > 2 and thdata.shape[2] > 1:
                        # Convert to numpy arrays and do clipping
                        lo_channel = np.clip(thdata[..., 1].astype(np.int32), 0, 255)
                        hi_channel = np.clip(thdata[..., 0].astype(np.int32), 0, 255)
                        
                        lomax = int(lo_channel.max())
                        posmax = lo_channel.argmax()
                        # Fix coordinate calculation
                        mcol = int(posmax // width)
                        mrow = int(posmax % width)
                        himax = int(hi_channel[mcol, mrow])
                        
                        maxtemp = ((himax + (lomax * 256)) / 64) - 273.15
                        maxtemp = round(maxtemp, 2)
                    else:
                        maxtemp = temp + 5
                        mcol = mrow = 64
                except Exception as e:
                    print(f"Error calculating maximum temperature: {e}")
                    maxtemp = temp + 5
                    mcol = mrow = 64
                
                # Minimum temperature  
                try:
                    if len(thdata.shape) > 2 and thdata.shape[2] > 1:
                        lo_channel = np.clip(thdata[..., 1].astype(np.int32), 0, 255)
                        hi_channel = np.clip(thdata[..., 0].astype(np.int32), 0, 255)
                        
                        lomin = int(lo_channel.min())
                        posmin = lo_channel.argmin()
                        # Fix coordinate calculation
                        lcol = int(posmin // width)
                        lrow = int(posmin % width)
                        himin = int(hi_channel[lcol, lrow])
                        
                        mintemp = ((himin + (lomin * 256)) / 64) - 273.15
                        mintemp = round(mintemp, 2)
                    else:
                        mintemp = temp - 5
                        lcol = lrow = 64
                except Exception as e:
                    print(f"Error calculating minimum temperature: {e}")
                    mintemp = temp - 5
                    lcol = lrow = 64
                
                # Average temperature
                try:
                    if len(thdata.shape) > 2 and thdata.shape[2] > 1:
                        lo_channel = np.clip(thdata[..., 1].astype(np.int32), 0, 255)
                        hi_channel = np.clip(thdata[..., 0].astype(np.int32), 0, 255)
                        
                        loavg = float(lo_channel.mean())
                        hiavg = float(hi_channel.mean())
                        
                        avgtemp = ((hiavg + (loavg * 256)) / 64) - 273.15
                        avgtemp = round(avgtemp, 2)
                    else:
                        avgtemp = temp
                except Exception as e:
                    print(f"Error calculating average temperature: {e}")
                    avgtemp = temp
            else:
                # Default values if no thermal data
                temp = maxtemp = mintemp = avgtemp = 20.0
                mcol = mrow = lcol = lrow = 64
            
            # Convert image to BGR
            try:
                if len(imdata.shape) == 3 and imdata.shape[2] >= 2:
                    bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
                else:
                    # If not YUV, try to use directly or convert from grayscale
                    if len(imdata.shape) == 2:
                        bgr = cv2.cvtColor(imdata, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr = imdata.copy()
            except cv2.error as e:
                print(f"Error converting color: {e}")
                # Fallback: use grayscale image
                if len(imdata.shape) == 3:
                    bgr = cv2.cvtColor(cv2.cvtColor(imdata, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                else:
                    bgr = cv2.cvtColor(imdata, cv2.COLOR_GRAY2BGR)
            
            # Apply contrast
            bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
            
            # Resize and apply blur
            bgr = cv2.resize(bgr, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
            if rad > 0:
                bgr = cv2.blur(bgr, (rad, rad))
            
            # Apply colormap
            colormaps = [
                (cv2.COLORMAP_JET, 'Jet'),
                (cv2.COLORMAP_HOT, 'Hot'),
                (cv2.COLORMAP_MAGMA, 'Magma'),
                (cv2.COLORMAP_INFERNO, 'Inferno'),
                (cv2.COLORMAP_PLASMA, 'Plasma'),
                (cv2.COLORMAP_BONE, 'Bone'),
                (cv2.COLORMAP_SPRING, 'Spring'),
                (cv2.COLORMAP_AUTUMN, 'Autumn'),
                (cv2.COLORMAP_VIRIDIS, 'Viridis'),
                (cv2.COLORMAP_PARULA, 'Parula'),
                (cv2.COLORMAP_RAINBOW, 'Rainbow')
            ]
            
            cmap, cmapText = colormaps[colormap % len(colormaps)]
            heatmap = cv2.applyColorMap(bgr, cmap)
            
            if colormap == 10:  # Inverted rainbow
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Draw center crosshairs
            center_x, center_y = newWidth // 2, newHeight // 2
            cv2.line(heatmap, (center_x, center_y + 20), (center_x, center_y - 20), (255, 255, 255), 2)
            cv2.line(heatmap, (center_x + 20, center_y), (center_x - 20, center_y), (255, 255, 255), 2)
            cv2.line(heatmap, (center_x, center_y + 20), (center_x, center_y - 20), (0, 0, 0), 1)
            cv2.line(heatmap, (center_x + 20, center_y), (center_x - 20, center_y), (0, 0, 0), 1)
            
            # Show central temperature
            cv2.putText(heatmap, f'{temp} C', (center_x + 10, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f'{temp} C', (center_x + 10, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Draw measurement points
            draw_measurement_points(heatmap, thdata, scale)
            
            # Main HUD
            if hud:
                cv2.rectangle(heatmap, (0, 0), (160, 120), (0, 0, 0), -1)
                texts = [
                    f'Avg Temp: {avgtemp} C',
                    f'Label Threshold: {threshold} C',
                    f'Colormap: {cmapText}',
                    f'Blur: {rad}',
                    f'Scaling: {scale}',
                    f'Contrast: {alpha}',
                    f'Snapshot: {snaptime}',
                    f'Recording: {elapsed}' + (' (REC)' if recording else '')
                ]
                
                for i, text in enumerate(texts):
                    color = (40, 40, 255) if recording and i == 7 else (0, 255, 255)
                    cv2.putText(heatmap, text, (10, 14 + i * 14),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
            # Measurement points HUD
            draw_measurement_hud(heatmap, thdata)
            
            # Show extreme temperatures
            if maxtemp > avgtemp + threshold:
                try:
                    pos_x, pos_y = min(mrow * scale, newWidth - 50), min(mcol * scale, newHeight - 20)
                    cv2.circle(heatmap, (pos_x, pos_y), 5, (0, 0, 0), 2)
                    cv2.circle(heatmap, (pos_x, pos_y), 5, (0, 0, 255), -1)
                    cv2.putText(heatmap, f'{maxtemp} C', (pos_x + 10, pos_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(heatmap, f'{maxtemp} C', (pos_x + 10, pos_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                except:
                    pass
            
            if mintemp < avgtemp - threshold:
                try:
                    pos_x, pos_y = min(lrow * scale, newWidth - 50), min(lcol * scale, newHeight - 20)
                    cv2.circle(heatmap, (pos_x, pos_y), 5, (0, 0, 0), 2)
                    cv2.circle(heatmap, (pos_x, pos_y), 5, (255, 0, 0), -1)
                    cv2.putText(heatmap, f'{mintemp} C', (pos_x + 10, pos_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(heatmap, f'{mintemp} C', (pos_x + 10, pos_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                except:
                    pass
            
            # Save data every second
            current_time = time.time()
            if current_time - last_save_time >= capture_interval:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                frame_filename = f"frame_{int(current_time)}.png"
                frame_filepath = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_filepath, heatmap)  # Save frame to folder

                save_to_csv(
                    timestamp=timestamp,
                    max_temp=maxtemp,
                    max_coords=(mcol, mrow),
                    min_temp=mintemp,
                    min_coords=(lcol, lrow),
                    avg_temp=avgtemp,
                    frame_file=frame_filepath
                )
                print(f"Data saved: {timestamp}, Max: {maxtemp}C, Min: {mintemp}C, Avg: {avgtemp}C")
                last_save_time = current_time
            
            # Show image
            cv2.imshow('Thermal', heatmap)
            
            # Recording handling
            if recording:
                if 'start' in locals():
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
                    if 'videoOut' in locals():
                        videoOut.write(heatmap)
            
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue
        
        # Key handling
        keyPress = cv2.waitKey(1) & 0xFF    
        
        if keyPress == ord('a'):
            rad += 1
        elif keyPress == ord('z'):
            rad = max(0, rad - 1)
        elif keyPress == ord('s'):
            threshold += 1
        elif keyPress == ord('x'):
            threshold = max(0, threshold - 1)
        elif keyPress == ord('d'):
            scale = min(5, scale + 1)
            newWidth, newHeight = width * scale, height * scale
            update_measurement_points_scale(scale)
            if not dispFullscreen and not isPi:
                cv2.resizeWindow('Thermal', newWidth, newHeight)
        elif keyPress == ord('c'):
            scale = max(1, scale - 1)
            newWidth, newHeight = width * scale, height * scale
            update_measurement_points_scale(scale)
            if not dispFullscreen and not isPi:
                cv2.resizeWindow('Thermal', newWidth, newHeight)
        elif keyPress == ord('q'):
            dispFullscreen = True
            cv2.namedWindow('Thermal', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Thermal', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif keyPress == ord('w'):
            dispFullscreen = False
            cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty('Thermal', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow('Thermal', newWidth, newHeight)
        elif keyPress == ord('f'):
            alpha = min(3.0, round(alpha + 0.1, 1))
        elif keyPress == ord('v'):
            alpha = max(0.0, round(alpha - 0.1, 1))
        elif keyPress == ord('h'):
            hud = not hud
        elif keyPress == ord('m'):
            colormap = (colormap + 1) % len(colormaps)
        elif keyPress == ord('n'):
            measurement_points.clear()
            print("Measurement points cleared")
        elif keyPress == ord('r') and not recording:
            now = time.strftime("%Y%m%d--%H%M%S")
            try:
                videoOut = cv2.VideoWriter(f'{now}output.avi', 
                                         cv2.VideoWriter_fourcc(*'XVID'), 25, 
                                         (newWidth, newHeight))
                recording = True
                start = time.time()
                print("Recording started")
            except Exception as e:
                print(f"Error starting recording: {e}")
        elif keyPress == ord('t') and recording:
            recording = False
            elapsed = "00:00:00"
            if 'videoOut' in locals():
                videoOut.release()
            print("Recording stopped")
        elif keyPress == ord('p'):
            now = time.strftime("%Y%m%d-%H%M%S")
            snaptime = time.strftime("%H:%M:%S")
            try:
                # Create snapshots directory if it doesn't exist
                snapshots_dir = "snapshots"
                os.makedirs(snapshots_dir, exist_ok=True)
                
                snapshot_path = os.path.join(snapshots_dir, f"TC001{now}.png")
                cv2.imwrite(snapshot_path, heatmap)
                print(f"Snapshot saved: {snapshot_path}")
                
                # Also save measurement points data
                if measurement_points:
                    measurements_path = os.path.join(snapshots_dir, f"TC001{now}_measurements.txt")
                    with open(measurements_path, "w") as f:
                        f.write(f"Snapshot taken: {snaptime}\n")
                        f.write(f"Average temperature: {avgtemp}C\n")
                        f.write(f"Central temperature: {temp}C\n")
                        f.write("Measurement points:\n")
                        for i, (thermal_x, thermal_y, _, _) in enumerate(measurement_points):
                            point_temp = get_temperature_at_point(thdata, thermal_x, thermal_y)
                            f.write(f"  P{i+1}: {point_temp}C at ({thermal_x}, {thermal_y})\n")
                    print(f"Measurement data saved: {measurements_path}")
            except Exception as e:
                print(f"Error saving snapshot: {e}")
        elif keyPress == 27:  # ESC
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if recording and 'videoOut' in locals():
        videoOut.release()
    print("Program finished")

if __name__ == "__main__":
    main()
