# FALLING-GEM

This repository contains the core logic for the "FALLING-GEM" interactive art installation, seamlessly bridging computer vision algorithms with physical hardware actuation.

## System Architecture

* **The Brain (Python):** Utilizes MediaPipe for real-time human pose tracking, dynamic spatial normalization, and OSC signal routing to TouchDesigner.
* **The Muscle (Arduino C++):** Handles asynchronous serial parsing and dual-threshold ultrasonic sensor debouncing to drive the physical servo mechanism.

## File Structure

* `main_vision_logic.py`: The core state machine and computer vision script.
* `arduino_hardware_control/arduino_hardware_control.ino`: The non-blocking hardware control logic.
* `requirements.txt`: Python environment dependencies.

## Installation & Setup

1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
