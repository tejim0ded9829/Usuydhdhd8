# advanced_captcha_solver.py
# Advanced AI-Powered hCaptcha Solver with YOLOv8 - Oct 2025 (LATEST)
# Author: Tejv
# Features: 11 captcha types + auto-save failed spot_different captchas

import os
import time
import base64
import random
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, List
from datetime import datetime
from collections import Counter
from selenium import webdriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pytesseract
from rich.console import Console
from captcha_patterns import CaptchaPatterns

console = Console()

class AdvancedCaptchaSolver:
    """
    Advanced hCaptcha solver with YOLOv8 + multi-method detection
    Supports 11 Discord captcha types (Oct 2025):
    
    1. Checkbox
    2. Drag Letter
    3. Select Smaller
    4. Drag Food
    5. Select Sitting
    6. Select Drinking
    7. Spot Different
    8. Scene Match
    9. Tool Cut
    10. Identical Elements
    11. Fit Inside
    """
    
    def __init__(self):
        self.browser = None
        self.yolo_model = self._load_yolo_model()
        self.action_chains = None
        self.max_attempts = 3
        self.solve_count = 0
        
        # Create fails directory for spot_different screenshots
        self.fails_dir = "fails"
        if not os.path.exists(self.fails_dir):
            os.makedirs(self.fails_dir)
            console.print(f"[cyan]✓ Created '{self.fails_dir}' directory for failed captchas[/cyan]")
    
    def _load_yolo_model(self):
        """Load YOLOv8 model for object detection"""
        try:
            from ultralytics import YOLO
            console.print("[cyan]Loading YOLOv8 model...[/cyan]")
            model = YOLO('yolov8n.pt')
            model.conf = 0.30
            model.iou = 0.45
            model.max_det = 10
            console.print("[green]✓ YOLOv8 loaded successfully[/green]")
            return model
        except Exception as e:
            console.print(f"[yellow]YOLOv8 unavailable: {e}[/yellow]")
            console.print("[yellow]Attempting YOLOv5 fallback...[/yellow]")
            try:
                import torch
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)
                model.conf = 0.35
                model.iou = 0.45
                console.print("[green]✓ YOLOv5 loaded as fallback[/green]")
                return model
            except:
                console.print("[yellow]⚠ No YOLO model available, using fallback detection[/yellow]")
                return None
    
    def _initialize_browser(self):
        """Initialize stealth browser with anti-detection"""
        try:
            console.print("[cyan]Initializing stealth browser...[/cyan]")
            
            options = webdriver.ChromeOptions()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            self.browser = uc.Chrome(options=options)
            self.action_chains = ActionChains(self.browser)
            
            # Anti-detection scripts
            self.browser.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    window.chrome = {runtime: {}};
                '''
            })
            
            console.print("[green]✓ Stealth browser initialized[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Browser initialization failed: {e}[/red]")
            return False
    
    def _preprocess_image(self, b64_img: str) -> Optional[np.ndarray]:
        """Preprocess base64 image for analysis"""
        try:
            if ',' in b64_img:
                b64_img = b64_img.split(',')[1]
            
            img_data = base64.b64decode(b64_img)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            console.print(f"[yellow]Image preprocessing error: {e}[/yellow]")
            return None
    
    def _analyze_with_yolo(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Analyze image using YOLO model"""
        if self.yolo_model is None or image is None:
            return None, 0.0
        
        try:
            # YOLOv8
            if hasattr(self.yolo_model, 'predict'):
                results = self.yolo_model.predict(image, verbose=False)
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        confidences = boxes.conf.cpu().numpy()
                        best_idx = confidences.argmax()
                        label_idx = int(boxes.cls[best_idx].item())
                        confidence = confidences[best_idx]
                        label = results[0].names[label_idx]
                        return label, float(confidence)
            # YOLOv5
            else:
                results = self.yolo_model(image)
                if results and len(results.xyxy[0]) > 0:
                    detections = results.pandas().xyxy[0]
                    if not detections.empty:
                        best_detection = detections.loc[detections['confidence'].idxmax()]
                        return best_detection['name'], best_detection['confidence']
            
            return None, 0.0
        except Exception as e:
            console.print(f"[yellow]YOLO analysis error: {e}[/yellow]")
            return None, 0.0
    
    def _get_image_size(self, image: np.ndarray) -> int:
        """Calculate approximate object size from image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                return int(area)
            
            return 0
        except:
            return 0
    
    def _extract_text_ocr(self, image: np.ndarray) -> Optional[str]:
        """Extract text from image using OCR"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_img = Image.fromarray(binary)
            text = pytesseract.image_to_string(
                pil_img,
                config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            ).strip()
            return text if text else None
        except Exception as e:
            console.print(f"[yellow]OCR error: {e}[/yellow]")
            return None
    
    def _get_dominant_color(self, image: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Get dominant color from image (for spot_different)"""
        try:
            pixels = np.float32(image.reshape(-1, 3))
            n_colors = 1
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            dominant = palette[np.argmax(counts)]
            return tuple(dominant.astype(int))
        except Exception as e:
            console.print(f"[yellow]Color analysis error: {e}[/yellow]")
            return None
    
    def _calculate_circularity(self, image: np.ndarray) -> float:
        """Calculate circularity of object in image (for spot_different shape analysis)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    return circularity
            
            return 0.0
        except Exception as e:
            console.print(f"[yellow]Circularity calculation error: {e}[/yellow]")
            return 0.0
    
    def _save_failed_captcha(self, captcha_type: str, additional_info: str = ""):
        """Save screenshot of failed captcha to fails/ directory"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{captcha_type}_{timestamp}{additional_info}.png"
            filepath = os.path.join(self.fails_dir, filename)
            
            screenshot = self.browser.get_screenshot_as_png()
            with open(filepath, 'wb') as f:
                f.write(screenshot)
            
            console.print(f"[yellow]Failed captcha saved: {filepath}[/yellow]")
            return filepath
        except Exception as e:
            console.print(f"[red]Failed to save screenshot: {e}[/red]")
            return None
    
    def _get_image_from_element(self, element) -> Optional[str]:
        """Extract base64 image from element"""
        try:
            return element.screenshot_as_base64
        except Exception as e:
            console.print(f"[yellow]Image extraction error: {e}[/yellow]")
            return None
    
    def _get_captcha_token(self) -> Optional[str]:
        """Extract hCaptcha response token"""
        try:
            token = self.browser.execute_script(
                "return document.querySelector('[name=\"h-captcha-response\"]')?.value || "
                "document.getElementById('h-captcha-response')?.value"
            )
            return token if token else None
        except:
            return None
    
    def solve_checkbox(self) -> Optional[str]:
        """Solve checkbox captcha"""
        try:
            console.print("[cyan]Solving checkbox...[/cyan]")
            
            selectors = [".checkbox", "#checkbox", "input[type='checkbox']", "[role='checkbox']"]
            checkbox = None
            
            for selector in selectors:
                try:
                    checkbox = WebDriverWait(self.browser, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    break
                except:
                    continue
            
            if checkbox:
                time.sleep(random.uniform(0.8, 1.5))
                checkbox.click()
                time.sleep(2.5)
                
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Checkbox solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[yellow]Checkbox error: {e}[/yellow]")
        
        return None
    
    def solve_fit_inside(self) -> Optional[str]:
        """Solve 'fit inside' captcha - NEW CAPTCHA TYPE Oct 2025"""
        try:
            console.print("[cyan]Solving fit inside captcha...[/cyan]")
            time.sleep(1.5)
            
            # Find reference item (the container - basket, box, etc.)
            reference_item = None
            reference_label = None
            reference_size = 0
            
            try:
                # Look for reference item indicator
                reference_selectors = [
                    '.sample-item', '.reference-item', '.example-image',
                    '.task-image:first-child', '.header-image', '.prompt img'
                ]
                
                for selector in reference_selectors:
                    try:
                        reference_item = self.browser.find_element(By.CSS_SELECTOR, selector)
                        break
                    except:
                        continue
                
                if reference_item:
                    ref_b64 = self._get_image_from_element(reference_item)
                    if ref_b64:
                        ref_image = self._preprocess_image(ref_b64)
                        if ref_image is not None:
                            reference_size = self._get_image_size(ref_image)
                            reference_label, _ = self._analyze_with_yolo(ref_image)
                            console.print(f"[green]Reference item: {reference_label} (size: {reference_size})[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not detect reference item: {e}[/yellow]")
            
            # Find all candidate items
            items = self.browser.find_elements(By.CSS_SELECTOR, 
                '.task-image .image, .challenge-view .image, [role="button"]')
            
            selected_count = 0
            
            # HTML label approach
            for item in items:
                try:
                    aria_label = (item.get_attribute('aria-label') or '').lower()
                    alt_text = (item.get_attribute('alt') or '').lower()
                    combined = aria_label + ' ' + alt_text
                    
                    # Check if item would fit in reference (basket, box, etc.)
                    if CaptchaPatterns.can_fit_inside(combined, reference_label):
                        item.click()
                        selected_count += 1
                        console.print(f"[green]Selected item that fits inside (HTML)[/green]")
                        time.sleep(random.uniform(0.3, 0.5))
                except:
                    continue
            
            # YOLO + size comparison approach
            if selected_count == 0 and self.yolo_model:
                console.print("[cyan]Using YOLO + size comparison...[/cyan]")
                
                for item in items:
                    try:
                        b64_img = self._get_image_from_element(item)
                        if not b64_img:
                            continue
                        
                        image = self._preprocess_image(b64_img)
                        if image is None:
                            continue
                        
                        label, confidence = self._analyze_with_yolo(image)
                        item_size = self._get_image_size(image)
                        
                        if label and confidence > 0.20:
                            # Check if item can fit based on category and size
                            if CaptchaPatterns.can_fit_inside(label, reference_label):
                                # Size validation: item should be smaller than reference
                                if reference_size == 0 or item_size < reference_size * 1.5:
                                    item.click()
                                    selected_count += 1
                                    console.print(f"[green]Selected: {label} ({confidence:.2f}, size: {item_size})[/green]")
                                    time.sleep(random.uniform(0.3, 0.5))
                    except:
                        continue
            
            if selected_count > 0:
                time.sleep(1.5)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Fit inside solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Fit inside error: {e}[/red]")
        
        return None
    
    def solve_spot_different(self) -> Optional[str]:
        """Solve spot the different object (color + shape analysis, auto-save fails)"""
        try:
            console.print("[cyan]Solving spot different object...[/cyan]")
            time.sleep(1.5)
            
            items = self.browser.find_elements(By.CSS_SELECTOR, '.task-image .image, .challenge-view .image, [role="button"], .ball, .object')
            
            if len(items) < 2:
                console.print("[yellow]Not enough items to compare[/yellow]")
                self._save_failed_captcha("spot_different", "_not_enough_items")
                return None
            
            # Collect color AND shape data
            analysis_data = []
            for item in items:
                try:
                    b64_img = self._get_image_from_element(item)
                    if not b64_img:
                        continue
                    
                    image = self._preprocess_image(b64_img)
                    if image is None:
                        continue
                    
                    dominant_color = self._get_dominant_color(image)
                    circularity = self._calculate_circularity(image)
                    
                    if dominant_color:
                        analysis_data.append({
                            'element': item,
                            'color': dominant_color,
                            'circularity': circularity
                        })
                except:
                    continue
            
            if len(analysis_data) < 2:
                console.print("[yellow]Could not analyze objects[/yellow]")
                self._save_failed_captcha("spot_different", "_analysis_failed")
                return None
            
            # Method 1: Find color outlier
            color_counts = Counter([str(d['color']) for d in analysis_data])
            for data in analysis_data:
                if color_counts[str(data['color'])] == 1:
                    console.print(f"[green]✓ Different object found (color): {data['color']}[/green]")
                    data['element'].click()
                    time.sleep(1.5)
                    try:
                        submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                        submit_btn.click()
                    except:
                        pass
                    time.sleep(3)
                    token = self._get_captcha_token()
                    if token:
                        console.print("[green]✓ Spot different solved (color)[/green]")
                        return token
            
            # Method 2: Find shape outlier (circularity)
            circularities = [d['circularity'] for d in analysis_data]
            avg_circularity = np.mean(circularities)
            max_deviation = 0
            outlier_data = None
            
            for data in analysis_data:
                deviation = abs(data['circularity'] - avg_circularity)
                if deviation > max_deviation:
                    max_deviation = deviation
                    outlier_data = data
            
            if outlier_data and max_deviation > 0.1:
                console.print(f"[green]✓ Different object found (shape): circularity {outlier_data['circularity']:.2f}[/green]")
                outlier_data['element'].click()
                time.sleep(1.5)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Spot different solved (shape)[/green]")
                    return token
            
            console.print("[yellow]⚠ Could not identify different object[/yellow]")
            self._save_failed_captcha("spot_different", "_no_outlier_found")
        
        except Exception as e:
            console.print(f"[red]Spot different error: {e}[/red]")
            self._save_failed_captcha("spot_different", f"_exception")
        
        return None
    
    def solve_drag_letter(self) -> Optional[str]:
        """Solve drag letter puzzle with OCR"""
        try:
            console.print("[cyan]Solving drag letter puzzle...[/cyan]")
            time.sleep(1.5)
            
            draggable = self.browser.find_element(By.CSS_SELECTOR, '[draggable="true"], .draggable, .move-handle')
            slots = self.browser.find_elements(By.CSS_SELECTOR, '.slot, .drop-zone, .letter-slot, .challenge-view .image')
            
            # OCR matching
            try:
                drag_b64 = self._get_image_from_element(draggable)
                drag_image = self._preprocess_image(drag_b64)
                drag_letter = self._extract_text_ocr(drag_image)
                
                if drag_letter:
                    console.print(f"[green]Source letter: {drag_letter}[/green]")
                    
                    for slot in slots:
                        slot_b64 = self._get_image_from_element(slot)
                        slot_image = self._preprocess_image(slot_b64)
                        slot_letter = self._extract_text_ocr(slot_image)
                        
                        if slot_letter and slot_letter.upper() == drag_letter.upper():
                            console.print(f"[green]✓ Matching slot: {slot_letter}[/green]")
                            time.sleep(0.5)
                            self.action_chains.click_and_hold(draggable).pause(0.4).move_to_element(slot).pause(0.3).release().perform()
                            time.sleep(2)
                            try:
                                submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                                submit_btn.click()
                            except:
                                pass
                            time.sleep(3)
                            token = self._get_captcha_token()
                            if token:
                                console.print("[green]✓ Drag letter solved (OCR)[/green]")
                                return token
            except Exception as e:
                console.print(f"[yellow]OCR matching failed: {e}[/yellow]")
            
            # Fallback: drag to middle slot
            if draggable and slots:
                target_slot = slots[len(slots) // 2] if len(slots) > 1 else slots[0]
                console.print("[yellow]Using fallback slot selection[/yellow]")
                time.sleep(0.5)
                self.action_chains.click_and_hold(draggable).pause(0.4).move_to_element(target_slot).pause(0.3).release().perform()
                time.sleep(2)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Drag letter solved (fallback)[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Drag letter error: {e}[/red]")
        
        return None
    
    def solve_select_smaller(self) -> Optional[str]:
        """Solve select smaller objects"""
        try:
            console.print("[cyan]Solving select smaller objects...[/cyan]")
            time.sleep(1.5)
            
            items = self.browser.find_elements(By.CSS_SELECTOR, '.task-image .image, .challenge-view .image, [role="button"]')
            selected_count = 0
            
            # HTML label detection
            for item in items:
                try:
                    aria_label = (item.get_attribute('aria-label') or '').lower()
                    alt_text = (item.get_attribute('alt') or '').lower()
                    combined = aria_label + ' ' + alt_text
                    
                    if CaptchaPatterns.is_small_object(combined):
                        item.click()
                        selected_count += 1
                        console.print(f"[green]Selected small object (HTML)[/green]")
                        time.sleep(random.uniform(0.3, 0.5))
                except:
                    continue
            
            # YOLO detection
            if selected_count == 0 and self.yolo_model:
                console.print("[cyan]Using YOLO detection...[/cyan]")
                
                for item in items:
                    try:
                        b64_img = self._get_image_from_element(item)
                        if not b64_img:
                            continue
                        
                        image = self._preprocess_image(b64_img)
                        if image is None:
                            continue
                        
                        label, confidence = self._analyze_with_yolo(image)
                        if label and CaptchaPatterns.is_small_object(label) and confidence > 0.20:
                            item.click()
                            selected_count += 1
                            console.print(f"[green]Selected: {label} ({confidence:.2f})[/green]")
                            time.sleep(random.uniform(0.3, 0.5))
                    except:
                        continue
            
            if selected_count > 0:
                time.sleep(1.5)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Select smaller solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Select smaller error: {e}[/red]")
        
        return None
    
    def solve_drag_food_to_animal(self) -> Optional[str]:
        """Solve drag food to animal"""
        try:
            console.print("[cyan]Solving drag food to animal...[/cyan]")
            time.sleep(1.5)
            
            items = self.browser.find_elements(By.CSS_SELECTOR, '.task-image .image, .challenge-view .image, [role="img"]')
            food_element = None
            animal_element = None
            
            # HTML detection
            for item in items:
                try:
                    aria_label = (item.get_attribute('aria-label') or '').lower()
                    alt_text = (item.get_attribute('alt') or '').lower()
                    combined = aria_label + ' ' + alt_text
                    
                    if CaptchaPatterns.is_food(combined):
                        food_element = item
                    if CaptchaPatterns.is_animal(combined):
                        animal_element = item
                except:
                    continue
            
            # YOLO fallback
            if (not food_element or not animal_element) and self.yolo_model:
                console.print("[cyan]Using YOLO detection...[/cyan]")
                
                for item in items:
                    try:
                        b64_img = self._get_image_from_element(item)
                        if not b64_img:
                            continue
                        
                        image = self._preprocess_image(b64_img)
                        if image is None:
                            continue
                        
                        label, confidence = self._analyze_with_yolo(image)
                        if label and confidence > 0.25:
                            if CaptchaPatterns.is_food(label) and not food_element:
                                food_element = item
                                console.print(f"[green]Food: {label} ({confidence:.2f})[/green]")
                            if CaptchaPatterns.is_animal(label) and not animal_element:
                                animal_element = item
                                console.print(f"[green]Animal: {label} ({confidence:.2f})[/green]")
                    except:
                        continue
            
            if food_element and animal_element:
                time.sleep(0.5)
                self.action_chains.click_and_hold(food_element).pause(0.4).move_to_element(animal_element).pause(0.3).release().perform()
                console.print("[green]✓ Drag completed[/green]")
                time.sleep(2)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Drag food solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Drag food error: {e}[/red]")
        
        return None
    
    def solve_select_sitting(self) -> Optional[str]:
        """Solve select sitting objects (includes towels)"""
        try:
            console.print("[cyan]Solving select sitting objects...[/cyan]")
            time.sleep(1.5)
            
            items = self.browser.find_elements(By.CSS_SELECTOR, '.task-image .image, .challenge-view .image, [role="button"]')
            selected_count = 0
            
            for item in items:
                try:
                    aria_label = (item.get_attribute('aria-label') or '').lower()
                    alt_text = (item.get_attribute('alt') or '').lower()
                    combined = aria_label + ' ' + alt_text
                    
                    if CaptchaPatterns.is_sitting_object(combined):
                        item.click()
                        selected_count += 1
                        console.print(f"[green]Selected sitting object[/green]")
                        time.sleep(random.uniform(0.3, 0.5))
                except:
                    continue
            
            # YOLO fallback
            if selected_count == 0 and self.yolo_model:
                console.print("[cyan]Using YOLO detection...[/cyan]")
                
                for item in items:
                    try:
                        b64_img = self._get_image_from_element(item)
                        if not b64_img:
                            continue
                        
                        image = self._preprocess_image(b64_img)
                        if image is None:
                            continue
                        
                        label, confidence = self._analyze_with_yolo(image)
                        if label and CaptchaPatterns.is_sitting_object(label) and confidence > 0.25:
                            item.click()
                            selected_count += 1
                            console.print(f"[green]Selected: {label} ({confidence:.2f})[/green]")
                            time.sleep(random.uniform(0.3, 0.5))
                    except:
                        continue
            
            if selected_count > 0:
                time.sleep(1.5)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Select sitting solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Select sitting error: {e}[/red]")
        
        return None
    
    def solve_select_drinking(self) -> Optional[str]:
        """Solve select drinking objects"""
        try:
            console.print("[cyan]Solving select drinking objects...[/cyan]")
            time.sleep(1.5)
            
            items = self.browser.find_elements(By.CSS_SELECTOR, '.task-image .image, .challenge-view .image, [role="button"]')
            selected_count = 0
            
            for item in items:
                try:
                    aria_label = (item.get_attribute('aria-label') or '').lower()
                    alt_text = (item.get_attribute('alt') or '').lower()
                    combined = aria_label + ' ' + alt_text
                    
                    if CaptchaPatterns.is_drinking_object(combined):
                        item.click()
                        selected_count += 1
                        console.print(f"[green]Selected drinking object[/green]")
                        time.sleep(random.uniform(0.3, 0.5))
                except:
                    continue
            
            # YOLO fallback
            if selected_count == 0 and self.yolo_model:
                console.print("[cyan]Using YOLO detection...[/cyan]")
                
                for item in items:
                    try:
                        b64_img = self._get_image_from_element(item)
                        if not b64_img:
                            continue
                        
                        image = self._preprocess_image(b64_img)
                        if image is None:
                            continue
                        
                        label, confidence = self._analyze_with_yolo(image)
                        if label and CaptchaPatterns.is_drinking_object(label) and confidence > 0.25:
                            item.click()
                            selected_count += 1
                            console.print(f"[green]Selected: {label} ({confidence:.2f})[/green]")
                            time.sleep(random.uniform(0.3, 0.5))
                    except:
                        continue
            
            if selected_count > 0:
                time.sleep(1.5)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Select drinking solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Select drinking error: {e}[/red]")
        
        return None
    
    def solve_scene_match(self) -> Optional[str]:
        """Solve scene match captcha"""
        try:
            console.print("[cyan]Solving scene match...[/cyan]")
            time.sleep(1.5)
            
            items = self.browser.find_elements(By.CSS_SELECTOR, '.task-image .image, .challenge-view .image, [role="button"]')
            selected_count = 0
            
            for item in items:
                try:
                    aria_label = (item.get_attribute('aria-label') or '').lower()
                    alt_text = (item.get_attribute('alt') or '').lower()
                    combined = aria_label + ' ' + alt_text
                    
                    if CaptchaPatterns.is_scene_match(combined):
                        item.click()
                        selected_count += 1
                        console.print(f"[green]Selected scene match object[/green]")
                        time.sleep(random.uniform(0.3, 0.5))
                except:
                    continue
            
            # YOLO fallback
            if selected_count == 0 and self.yolo_model:
                console.print("[cyan]Using YOLO detection...[/cyan]")
                
                for item in items:
                    try:
                        b64_img = self._get_image_from_element(item)
                        if not b64_img:
                            continue
                        
                        image = self._preprocess_image(b64_img)
                        if image is None:
                            continue
                        
                        label, confidence = self._analyze_with_yolo(image)
                        if label and CaptchaPatterns.is_scene_match(label) and confidence > 0.20:
                            item.click()
                            selected_count += 1
                            console.print(f"[green]Selected: {label} ({confidence:.2f})[/green]")
                            time.sleep(random.uniform(0.3, 0.5))
                    except:
                        continue
            
            if selected_count > 0:
                time.sleep(1.5)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Scene match solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Scene match error: {e}[/red]")
        
        return None
    
    def solve_tool_cut(self) -> Optional[str]:
        """Solve tool cut captcha"""
        try:
            console.print("[cyan]Solving tool cut...[/cyan]")
            time.sleep(1.5)
            
            items = self.browser.find_elements(By.CSS_SELECTOR, '.task-image .image, .challenge-view .image, [role="button"]')
            selected_count = 0
            
            for item in items:
                try:
                    aria_label = (item.get_attribute('aria-label') or '').lower()
                    alt_text = (item.get_attribute('alt') or '').lower()
                    combined = aria_label + ' ' + alt_text
                    
                    if CaptchaPatterns.is_tool_cut(combined):
                        item.click()
                        selected_count += 1
                        console.print(f"[green]Selected cuttable object[/green]")
                        time.sleep(random.uniform(0.3, 0.5))
                except:
                    continue
            
            # YOLO fallback
            if selected_count == 0 and self.yolo_model:
                console.print("[cyan]Using YOLO detection...[/cyan]")
                
                for item in items:
                    try:
                        b64_img = self._get_image_from_element(item)
                        if not b64_img:
                            continue
                        
                        image = self._preprocess_image(b64_img)
                        if image is None:
                            continue
                        
                        label, confidence = self._analyze_with_yolo(image)
                        if label and CaptchaPatterns.is_tool_cut(label) and confidence > 0.20:
                            item.click()
                            selected_count += 1
                            console.print(f"[green]Selected: {label} ({confidence:.2f})[/green]")
                            time.sleep(random.uniform(0.3, 0.5))
                    except:
                        continue
            
            if selected_count > 0:
                time.sleep(1.5)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Tool cut solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Tool cut error: {e}[/red]")
        
        return None
    
    def solve_identical(self) -> Optional[str]:
        """Solve identical elements captcha"""
        try:
            console.print("[cyan]Solving identical elements...[/cyan]")
            time.sleep(1.5)
            
            items = self.browser.find_elements(By.CSS_SELECTOR, '.task-image .image, .challenge-view .image, [role="button"]')
            
            # Extract visual features and find duplicates
            item_data = []
            for item in items:
                try:
                    b64_img = self._get_image_from_element(item)
                    if not b64_img:
                        continue
                    
                    image = self._preprocess_image(b64_img)
                    if image is None:
                        continue
                    
                    # Use image hash for comparison
                    img_hash = hash(image.tobytes())
                    item_data.append({'element': item, 'hash': img_hash})
                except:
                    continue
            
            # Find duplicates
            hash_counts = Counter([d['hash'] for d in item_data])
            clicked_count = 0
            
            for data in item_data:
                if hash_counts[data['hash']] >= 2 and clicked_count < 2:
                    data['element'].click()
                    clicked_count += 1
                    console.print(f"[green]Clicked identical element {clicked_count}/2[/green]")
                    time.sleep(0.5)
                    if clicked_count == 2:
                        break
            
            if clicked_count == 2:
                time.sleep(1.5)
                try:
                    submit_btn = self.browser.find_element(By.CSS_SELECTOR, "button[type='submit'], .button--verify")
                    submit_btn.click()
                except:
                    pass
                time.sleep(3)
                token = self._get_captcha_token()
                if token:
                    console.print("[green]✓ Identical elements solved[/green]")
                    return token
        
        except Exception as e:
            console.print(f"[red]Identical elements error: {e}[/red]")
        
        return None
    
    def solve_captcha(self, site_key: str, site_url: str) -> Optional[str]:
        """Main entry point for captcha solving"""
        if self.browser is None:
            if not self._initialize_browser():
                return None
        
        try:
            console.print(f"[cyan]Navigating to captcha site...[/cyan]")
            self.browser.get(site_url)
            time.sleep(3)
            
            for attempt in range(self.max_attempts):
                try:
                    # Find prompt
                    prompt_selectors = [".prompt-text", ".challenge-header", "h2", ".question"]
                    prompt_element = None
                    
                    for selector in prompt_selectors:
                        try:
                            prompt_element = WebDriverWait(self.browser, 5).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                            )
                            break
                        except:
                            continue
                    
                    if not prompt_element:
                        console.print("[yellow]No prompt found, trying checkbox[/yellow]")
                        token = self.solve_checkbox()
                        if token:
                            return token
                        continue
                    
                    prompt_text = prompt_element.text.lower()
                    console.print(f"[cyan]Prompt: {prompt_text[:60]}...[/cyan]")
                    
                    # Detect captcha type
                    captcha_type = CaptchaPatterns.detect_type(prompt_text)
                    console.print(f"[bold cyan]Detected type: {captcha_type}[/bold cyan]")
                    
                    # Route to appropriate solver (NOW INCLUDING fit_inside!)
                    if captcha_type == 'fit_inside':
                        token = self.solve_fit_inside()
                    elif captcha_type == 'spot_different':
                        token = self.solve_spot_different()
                    elif captcha_type == 'select_sitting':
                        token = self.solve_select_sitting()
                    elif captcha_type == 'select_drinking':
                        token = self.solve_select_drinking()
                    elif captcha_type == 'scene_match':
                        token = self.solve_scene_match()
                    elif captcha_type == 'tool_cut':
                        token = self.solve_tool_cut()
                    elif captcha_type == 'identical':
                        token = self.solve_identical()
                    elif captcha_type == 'drag_letter':
                        token = self.solve_drag_letter()
                    elif captcha_type == 'select_smaller':
                        token = self.solve_select_smaller()
                    elif captcha_type == 'drag_food':
                        token = self.solve_drag_food_to_animal()
                    else:
                        token = self.solve_checkbox()
                    
                    if token:
                        self.solve_count += 1
                        console.print(f"[bold green]✓ Captcha solved! (Total: {self.solve_count})[/bold green]")
                        return token
                    
                    console.print(f"[yellow]Attempt {attempt + 1}/{self.max_attempts} failed, retrying...[/yellow]")
                    time.sleep(2)
                
                except Exception as e:
                    console.print(f"[yellow]Detection error: {e}[/yellow]")
                    time.sleep(2)
            
            console.print("[red]All captcha solving attempts exhausted[/red]")
            return None
        
        except Exception as e:
            console.print(f"[red]Fatal captcha solve error: {e}[/red]")
            return None
    
    def close(self):
        """Close browser and cleanup resources"""
        if self.browser:
            try:
                # Clear cookies and cache before closing
                self.browser.delete_all_cookies()
                
                # Close all windows/tabs
                for handle in self.browser.window_handles:
                    self.browser.switch_to.window(handle)
                    self.browser.close()
                
                # Quit the browser instance
                self.browser.quit()
                console.print("[green]✓ Browser closed successfully[/green]")
            except Exception as e:
                console.print(f"[yellow]Browser cleanup warning: {e}[/yellow]")
            finally:
                self.browser = None
        
        # Clear YOLO model from memory if loaded
        if self.yolo_model is not None:
            try:
                del self.yolo_model
                self.yolo_model = None
                console.print("[green]✓ YOLO model unloaded from memory[/green]")
            except:
                pass
        
        # Reset action chains
        self.action_chains = None
        console.print("[cyan]✓ All resources cleaned up[/cyan]")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion"""
        try:
            self.close()
        except:
            pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.close()
        return False