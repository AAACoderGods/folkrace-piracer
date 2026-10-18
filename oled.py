"""
OLED Display module for PiRacer Pro.
Shows CPU, memory usage, and car speed on the onboard OLED screen.
"""

import logging
import time
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try importing luma.oled for SSD1306 OLED display
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    from luma.core.render import canvas
    LUMA_AVAILABLE = True
except ImportError:
    LUMA_AVAILABLE = False

# Try importing Adafruit SSD1306 as fallback
try:
    import board
    import adafruit_ssd1306
    ADAFRUIT_AVAILABLE = True
except ImportError:
    ADAFRUIT_AVAILABLE = False


class OLEDDisplay:
    """OLED display controller for PiRacer Pro."""

    def __init__(self, config):
        self.config = config
        self.device = None
        self.disp = None
        self.last_update_time = 0
        self.enabled = config.enable_oled

    def start(self):
        """Initialize OLED display."""
        if not self.enabled:
            logging.info("OLED display disabled in config.")
            return

        # Try luma.oled first
        if LUMA_AVAILABLE:
            try:
                serial = i2c(port=1, address=self.config.i2c_address)
                self.device = ssd1306(serial, width=self.config.width, height=self.config.height)
                logging.info(f"OLED: Initialized via luma.oled (I2C address: 0x{self.config.i2c_address:02X})")
                return
            except Exception as e:
                logging.warning(f"Could not initialize OLED via luma.oled: {e}")

        # Try Adafruit SSD1306 as fallback
        if ADAFRUIT_AVAILABLE:
            try:
                i2c = board.I2C()
                self.disp = adafruit_ssd1306.SSD1306_I2C(
                    self.config.width,
                    self.config.height,
                    i2c,
                    addr=self.config.i2c_address
                )
                self.disp.fill(0)
                self.disp.show()
                logging.info(f"OLED: Initialized via adafruit_ssd1306 (I2C address: 0x{self.config.i2c_address:02X})")
                return
            except Exception as e:
                logging.warning(f"Could not initialize OLED via adafruit_ssd1306: {e}")

        logging.warning("=" * 60)
        logging.warning("⚠️  NO OLED DISPLAY DETECTED!")
        logging.warning("   Display will be disabled.")
        logging.warning("   To enable OLED, ensure I2C is enabled and install:")
        logging.warning("   pip3 install luma.oled")
        logging.warning("   or")
        logging.warning("   pip3 install adafruit-circuitpython-ssd1306")
        logging.warning("=" * 60)
        self.enabled = False

    def get_system_stats(self) -> tuple:
        """Get CPU and memory usage percentages."""
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0

        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_percent = psutil.virtual_memory().percent
            return cpu_percent, mem_percent
        except Exception as e:
            logging.debug(f"Error reading system stats: {e}")
            return 0.0, 0.0

    def update(self, throttle: float):
        """Update OLED display with current stats."""
        if not self.enabled:
            return

        # Rate limiting: update at most every config.update_interval seconds
        current_time = time.time()
        if current_time - self.last_update_time < self.config.update_interval:
            return
        self.last_update_time = current_time

        cpu_percent, mem_percent = self.get_system_stats()
        
        # Calculate car speed (throttle as percentage)
        car_speed = abs(throttle) * 100

        # Format: CPU: %CPU%   MEM:%MEM%
        #         SPEED: %CARSPEED%
        line1 = f"CPU: {cpu_percent:4.1f}%   MEM:{mem_percent:3.0f}%"
        line2 = f"SPEED: {car_speed:5.1f}%"

        # Display using luma.oled
        if self.device is not None:
            try:
                with canvas(self.device) as draw:
                    # Clear display
                    draw.rectangle(self.device.bounding_box, outline=0, fill=0)
                    
                    # Draw text
                    from luma.core.util import font
                    draw.text((0, 0), line1, fill="white", font=font)
                    draw.text((0, 16), line2, fill="white", font=font)
            except Exception as e:
                logging.error(f"Error updating OLED via luma.oled: {e}")

        # Display using Adafruit SSD1306
        elif self.disp is not None:
            try:
                self.disp.fill(0)
                self.disp.text(line1, 0, 0, 1)
                self.disp.text(line2, 0, 16, 1)
                self.disp.show()
            except Exception as e:
                logging.error(f"Error updating OLED via adafruit_ssd1306: {e}")

    def stop(self):
        """Clear and stop OLED display."""
        if not self.enabled:
            return

        if self.device is not None:
            try:
                with canvas(self.device) as draw:
                    draw.rectangle(self.device.bounding_box, outline=0, fill=0)
            except Exception as e:
                logging.error(f"Error clearing OLED via luma.oled: {e}")

        elif self.disp is not None:
            try:
                self.disp.fill(0)
                self.disp.show()
            except Exception as e:
                logging.error(f"Error clearing OLED via adafruit_ssd1306: {e}")
