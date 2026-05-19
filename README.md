# AI Zero-Shot Surveillance System

An AI-powered real-time surveillance system that combines YOLOv8 object detection with CLIP-based zero-shot classification for intelligent activity monitoring and suspicious activity detection.

## Features

- Real-time surveillance video analysis
- YOLOv8 object detection
- CLIP zero-shot activity recognition
- Prompt-based classification
- Suspicious activity alerts
- Screenshot evidence capture
- Streamlit monitoring dashboard
- Optimized real-time inference pipeline

## Tech Stack

- Python
- PyTorch
- OpenCV
- YOLOv8
- CLIP
- Streamlit
- Transformers

## System Architecture

Video Feed → YOLO Detection → Object Cropping → CLIP Classification → Alert Generation → Dashboard Display

## Example Zero-Shot Labels

- person running
- person fighting
- suspicious activity
- person with weapon
- person stealing

## Installation

```bash
pip install -r requirements.txt