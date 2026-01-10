#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JetRacer Control Test Script

This script helps test and calibrate JetRacer's steering and throttle controls,
including the effects of gain parameters. It supports multiple test modes:
- Circle mode: Make the car drive in circles
- Step test: Test different values step by step
- Interactive mode: Adjust values in real-time

Python 3.6+ compatible
"""

from __future__ import print_function, division

import argparse
import time
import sys
import signal
import threading

# Python 3.6 compatible type hints
from typing import Optional  # Python 3.5+ (3.6 definitely has this)


class JetRacerActuator:
    """Simple wrapper for JetRacer control"""
    def __init__(self, throttle_gain: float = 1.0, steering_gain: float = 1.0, steering_offset: float = 0.0):
        self._car = None
        self._throttle_gain = throttle_gain
        self._steering_gain = steering_gain
        self._steering_offset = steering_offset
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self._car = NvidiaRacecar()
            self._car.throttle_gain = throttle_gain
            self._car.steering_gain = steering_gain
            self._car.steering_offset = steering_offset
            print(f"✓ JetRacer initialized with throttle_gain={throttle_gain}, steering_gain={steering_gain}, steering_offset={steering_offset}")
        except ImportError:
            print("⚠ Warning: 'jetracer' not found. Running in mock mode.")
            self._car = None
    
    def set_gains(self, throttle_gain: float, steering_gain: float, steering_offset: float = 0.0):
        """Update gain parameters"""
        self._throttle_gain = throttle_gain
        self._steering_gain = steering_gain
        self._steering_offset = steering_offset
        if self._car:
            self._car.throttle_gain = throttle_gain
            self._car.steering_gain = steering_gain
            self._car.steering_offset = steering_offset
    
    def apply(self, throttle: float, steering: float, log: bool = False):
        """Apply throttle and steering values"""
        if self._car:
            # Clip to valid ranges
            clipped_throttle = float(max(0.0, min(1.0, throttle)))
            clipped_steering = float(max(-1.0, min(1.0, steering)))
            self._car.throttle = clipped_throttle
            self._car.steering = clipped_steering
            if log:
                actual_throttle = clipped_throttle * self._throttle_gain
                actual_steering = clipped_steering * self._steering_gain + self._steering_offset
                print(f"  Input: throttle={clipped_throttle:.3f}, steering={clipped_steering:.3f} | "
                      f"Effective: throttle={actual_throttle:.3f}, steering={actual_steering:.3f}")
        elif log:
            print(f"  [Mock] throttle={throttle:.3f}, steering={steering:.3f}")
    
    def stop(self):
        """Stop the car"""
        if self._car:
            self._car.throttle = 0.0
            print("✓ Car stopped")


# Global shutdown flag
shutdown_flag = threading.Event()
actuator = None  # type: Optional[JetRacerActuator]


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\n\n[Signal] Shutting down...")
    shutdown_flag.set()


def cleanup():
    """Cleanup resources"""
    if actuator:
        actuator.stop()


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
import atexit
atexit.register(cleanup)


def test_circle_mode(args):
    """Test circle mode - make car drive in circles"""
    global actuator
    
    throttle = args.throttle
    steering = args.steering
    
    print("\n" + "="*70)
    print("CIRCLE MODE TEST")
    print("="*70)
    print(f"Throttle: {throttle:.3f} (gain: {args.throttle_gain:.3f})")
    print(f"Steering: {steering:.3f} (gain: {args.steering_gain:.3f}, offset: {args.steering_offset:.3f})")
    print(f"Duration: {args.duration:.1f}s")
    print(f"Log interval: {args.log_interval} frames")
    print("="*70)
    print("Starting in 3 seconds... (Press Ctrl+C to stop early)")
    print()
    
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
        if shutdown_flag.is_set():
            return
    
    print("\n✓ Starting circle test...\n")
    
    start_time = time.time()
    frame_count = 0
    dt = 1.0 / args.fps
    
    try:
        while not shutdown_flag.is_set():
            t0 = time.time()
            
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                print("\n⏱ Duration reached. Stopping...")
                break
            
            should_log = args.log_interval > 0 and (frame_count % args.log_interval == 0)
            
            if should_log:
                print(f"[t={elapsed:6.2f}s, frame={frame_count:5d}] ", end="")
            
            actuator.apply(throttle, steering, log=should_log)
            
            frame_count += 1
            
            # Maintain target FPS
            t_process = time.time() - t0
            if t_process < dt:
                time.sleep(dt - t_process)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    finally:
        actuator.stop()
        print(f"\n✓ Test completed. Total frames: {frame_count}, Duration: {time.time() - start_time:.2f}s")


def test_step_mode(args):
    """Test step mode - test different values step by step"""
    global actuator
    
    print("\n" + "="*70)
    print("STEP TEST MODE")
    print("="*70)
    print(f"Throttle range: {args.throttle_min:.3f} to {args.throttle_max:.3f} (step: {args.throttle_step:.3f})")
    print(f"Steering range: {args.steering_min:.3f} to {args.steering_max:.3f} (step: {args.steering_step:.3f})")
    print(f"Test duration per step: {args.step_duration:.1f}s")
    print(f"Gains: throttle_gain={args.throttle_gain:.3f}, steering_gain={args.steering_gain:.3f}, steering_offset={args.steering_offset:.3f}")
    print("="*70)
    print("Starting in 3 seconds... (Press Ctrl+C to stop)")
    print()
    
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
        if shutdown_flag.is_set():
            return
    
    print("\n✓ Starting step test...\n")
    
    dt = 1.0 / args.fps
    step_count = 0
    
    try:
        # Test throttle variations with fixed steering
        if args.test_throttle:
            print("\n" + "-"*70)
            print("Testing THROTTLE variations (steering fixed at {})".format(args.steering))
            print("-"*70)
            
            throttle_values = []
            t = args.throttle_min
            while t <= args.throttle_max + 0.0001:  # Add small epsilon for float comparison
                throttle_values.append(t)
                t += args.throttle_step
            
            for throttle in throttle_values:
                if shutdown_flag.is_set():
                    break
                
                print(f"\n[Step {step_count + 1}] Testing throttle={throttle:.3f}, steering={args.steering:.3f}")
                
                start_time = time.time()
                frame_count = 0
                
                while not shutdown_flag.is_set():
                    t0 = time.time()
                    elapsed = time.time() - start_time
                    
                    if elapsed >= args.step_duration:
                        break
                    
                    should_log = args.log_interval > 0 and (frame_count % args.log_interval == 0)
                    if should_log:
                        print(f"  [t={elapsed:5.2f}s] ", end="")
                    
                    actuator.apply(throttle, args.steering, log=should_log)
                    frame_count += 1
                    
                    t_process = time.time() - t0
                    if t_process < dt:
                        time.sleep(dt - t_process)
                
                actuator.stop()
                time.sleep(0.5)  # Brief pause between tests
                step_count += 1
        
        # Test steering variations with fixed throttle
        if args.test_steering:
            print("\n" + "-"*70)
            print("Testing STEERING variations (throttle fixed at {})".format(args.throttle))
            print("-"*70)
            
            steering_values = []
            s = args.steering_min
            while s <= args.steering_max + 0.0001:
                steering_values.append(s)
                s += args.steering_step
            
            for steering in steering_values:
                if shutdown_flag.is_set():
                    break
                
                print(f"\n[Step {step_count + 1}] Testing throttle={args.throttle:.3f}, steering={steering:.3f}")
                
                start_time = time.time()
                frame_count = 0
                
                while not shutdown_flag.is_set():
                    t0 = time.time()
                    elapsed = time.time() - start_time
                    
                    if elapsed >= args.step_duration:
                        break
                    
                    should_log = args.log_interval > 0 and (frame_count % args.log_interval == 0)
                    if should_log:
                        print(f"  [t={elapsed:5.2f}s] ", end="")
                    
                    actuator.apply(args.throttle, steering, log=should_log)
                    frame_count += 1
                    
                    t_process = time.time() - t0
                    if t_process < dt:
                        time.sleep(dt - t_process)
                
                actuator.stop()
                time.sleep(0.5)  # Brief pause between tests
                step_count += 1
        
        print("\n✓ Step test completed!")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    finally:
        actuator.stop()


def test_gain_mode(args):
    """Test gain parameters - test different gain values"""
    global actuator
    
    print("\n" + "="*70)
    print("GAIN TEST MODE")
    print("="*70)
    print(f"Fixed throttle: {args.throttle:.3f}")
    print(f"Fixed steering: {args.steering:.3f}")
    print(f"Throttle gain range: {args.throttle_gain_min:.3f} to {args.throttle_gain_max:.3f} (step: {args.throttle_gain_step:.3f})")
    print(f"Steering gain range: {args.steering_gain_min:.3f} to {args.steering_gain_max:.3f} (step: {args.steering_gain_step:.3f})")
    print(f"Test duration per step: {args.step_duration:.1f}s")
    print("="*70)
    print("Starting in 3 seconds... (Press Ctrl+C to stop)")
    print()
    
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
        if shutdown_flag.is_set():
            return
    
    print("\n✓ Starting gain test...\n")
    
    dt = 1.0 / args.fps
    step_count = 0
    
    try:
        # Test throttle gain variations
        if args.test_throttle_gain:
            print("\n" + "-"*70)
            print("Testing THROTTLE GAIN (throttle={}, steering={})".format(args.throttle, args.steering))
            print("-"*70)
            
            throttle_gain_values = []
            tg = args.throttle_gain_min
            while tg <= args.throttle_gain_max + 0.0001:
                throttle_gain_values.append(tg)
                tg += args.throttle_gain_step
            
            for throttle_gain in throttle_gain_values:
                if shutdown_flag.is_set():
                    break
                
                print(f"\n[Step {step_count + 1}] Testing throttle_gain={throttle_gain:.3f} "
                      f"(effective throttle: {args.throttle * throttle_gain:.3f})")
                
                actuator.set_gains(throttle_gain, args.steering_gain, args.steering_offset)
                
                start_time = time.time()
                frame_count = 0
                
                while not shutdown_flag.is_set():
                    t0 = time.time()
                    elapsed = time.time() - start_time
                    
                    if elapsed >= args.step_duration:
                        break
                    
                    should_log = args.log_interval > 0 and (frame_count % args.log_interval == 0)
                    if should_log:
                        print(f"  [t={elapsed:5.2f}s] ", end="")
                    
                    actuator.apply(args.throttle, args.steering, log=should_log)
                    frame_count += 1
                    
                    t_process = time.time() - t0
                    if t_process < dt:
                        time.sleep(dt - t_process)
                
                actuator.stop()
                time.sleep(0.5)
                step_count += 1
        
        # Test steering gain variations
        if args.test_steering_gain:
            print("\n" + "-"*70)
            print("Testing STEERING GAIN (throttle={}, steering={})".format(args.throttle, args.steering))
            print("-"*70)
            
            steering_gain_values = []
            sg = args.steering_gain_min
            while sg <= args.steering_gain_max + 0.0001:
                steering_gain_values.append(sg)
                sg += args.steering_gain_step
            
            for steering_gain in steering_gain_values:
                if shutdown_flag.is_set():
                    break
                
                effective_steering = args.steering * steering_gain + args.steering_offset
                print(f"\n[Step {step_count + 1}] Testing steering_gain={steering_gain:.3f} "
                      f"(effective steering: {effective_steering:.3f})")
                
                actuator.set_gains(args.throttle_gain, steering_gain, args.steering_offset)
                
                start_time = time.time()
                frame_count = 0
                
                while not shutdown_flag.is_set():
                    t0 = time.time()
                    elapsed = time.time() - start_time
                    
                    if elapsed >= args.step_duration:
                        break
                    
                    should_log = args.log_interval > 0 and (frame_count % args.log_interval == 0)
                    if should_log:
                        print(f"  [t={elapsed:5.2f}s] ", end="")
                    
                    actuator.apply(args.throttle, args.steering, log=should_log)
                    frame_count += 1
                    
                    t_process = time.time() - t0
                    if t_process < dt:
                        time.sleep(dt - t_process)
                
                actuator.stop()
                time.sleep(0.5)
                step_count += 1
        
        print("\n✓ Gain test completed!")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    finally:
        actuator.stop()


def main():
    global actuator
    
    parser = argparse.ArgumentParser(
        description="Test JetRacer steering and throttle controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Circle mode - make car drive in circles
  python test_jetracer_controls.py --mode circle --throttle 0.3 --steering 0.5 --duration 10

  # Step test - test different throttle values
  python test_jetracer_controls.py --mode step --test-throttle --throttle-min 0.1 --throttle-max 0.5 --throttle-step 0.1

  # Step test - test different steering values
  python test_jetracer_controls.py --mode step --test-steering --steering-min -0.5 --steering-max 0.5 --steering-step 0.2

  # Gain test - test throttle gain
  python test_jetracer_controls.py --mode gain --test-throttle-gain --throttle-gain-min 0.2 --throttle-gain-max 0.8 --throttle-gain-step 0.2

  # Gain test - test steering gain
  python test_jetracer_controls.py --mode gain --test-steering-gain --steering-gain-min 0.3 --steering-gain-max 1.0 --steering-gain-step 0.2
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["circle", "step", "gain"], default="circle",
                       help="Test mode: circle (drive in circles), step (test different values), gain (test gain parameters)")
    
    # Common parameters
    parser.add_argument("--throttle", type=float, default=0.3, help="Throttle value [0, 1]")
    parser.add_argument("--steering", type=float, default=0.5, help="Steering value [-1, 1], negative=left, positive=right")
    parser.add_argument("--throttle-gain", type=float, default=0.4, help="Throttle gain (multiplier)")
    parser.add_argument("--steering-gain", type=float, default=0.6, help="Steering gain (multiplier)")
    parser.add_argument("--steering-offset", type=float, default=0.0, help="Steering offset (added after gain)")
    parser.add_argument("--fps", type=float, default=20.0, help="Control loop FPS")
    parser.add_argument("--log-interval", type=int, default=20, help="Log every N frames (0 to disable)")
    
    # Circle mode parameters
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds (circle mode)")
    
    # Step test parameters
    parser.add_argument("--test-throttle", action="store_true", help="Test throttle variations (step mode)")
    parser.add_argument("--test-steering", action="store_true", help="Test steering variations (step mode)")
    parser.add_argument("--throttle-min", type=float, default=0.1, help="Minimum throttle value (step mode)")
    parser.add_argument("--throttle-max", type=float, default=0.5, help="Maximum throttle value (step mode)")
    parser.add_argument("--throttle-step", type=float, default=0.1, help="Throttle step size (step mode)")
    parser.add_argument("--steering-min", type=float, default=-0.5, help="Minimum steering value (step mode)")
    parser.add_argument("--steering-max", type=float, default=0.5, help="Maximum steering value (step mode)")
    parser.add_argument("--steering-step", type=float, default=0.2, help="Steering step size (step mode)")
    parser.add_argument("--step-duration", type=float, default=3.0, help="Duration per test step in seconds")
    
    # Gain test parameters
    parser.add_argument("--test-throttle-gain", action="store_true", help="Test throttle gain variations (gain mode)")
    parser.add_argument("--test-steering-gain", action="store_true", help="Test steering gain variations (gain mode)")
    parser.add_argument("--throttle-gain-min", type=float, default=0.2, help="Minimum throttle gain (gain mode)")
    parser.add_argument("--throttle-gain-max", type=float, default=1.0, help="Maximum throttle gain (gain mode)")
    parser.add_argument("--throttle-gain-step", type=float, default=0.2, help="Throttle gain step size (gain mode)")
    parser.add_argument("--steering-gain-min", type=float, default=0.3, help="Minimum steering gain (gain mode)")
    parser.add_argument("--steering-gain-max", type=float, default=1.0, help="Maximum steering gain (gain mode)")
    parser.add_argument("--steering-gain-step", type=float, default=0.2, help="Steering gain step size (gain mode)")
    
    args = parser.parse_args()
    
    # Initialize actuator
    actuator = JetRacerActuator(
        throttle_gain=args.throttle_gain,
        steering_gain=args.steering_gain,
        steering_offset=args.steering_offset
    )
    
    # Run test based on mode
    try:
        if args.mode == "circle":
            test_circle_mode(args)
        elif args.mode == "step":
            if not (args.test_throttle or args.test_steering):
                print("⚠ Error: Step mode requires --test-throttle or --test-steering")
                sys.exit(1)
            test_step_mode(args)
        elif args.mode == "gain":
            if not (args.test_throttle_gain or args.test_steering_gain):
                print("⚠ Error: Gain mode requires --test-throttle-gain or --test-steering-gain")
                sys.exit(1)
            test_gain_mode(args)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()


