# captcha_patterns.py
# Extended pattern support for Discord hCaptcha, Oct 2025 (ALL 11 types)

class CaptchaPatterns:
    """
    Discord hCaptcha types handled:
    1. Checkbox ("I am human")
    2. Drag Letter to Match (any letter, any slot)
    3. Select objects "normally smaller than reference item"
    4. Drag food to animal (chicken/turkey to bear)
    5. Pick the thing made for sitting on (chairs, stools, benches, towels)
    6. Identify all objects designed for drinking from (cups, glasses, mugs)
    7. Spot the different object (odd color/shape ball)
    8. Scene match/pick matching objects from example scene (e.g. apples in grocery context)
    9. Tool cut: pick objects the shown sample tool (scissors/knife) can cut
    10. Click the two elements that are identical (shapes/symbols)
    11. Fit inside: Click all objects that fit inside the sample item (glass, orange juice)
    """
    
    CHECKBOX_KEYWORDS = [
        'i am human', 'not a robot', 'are you human', 'wait! are you human', 'confirm you're not a robot'
    ]
    
    DRAG_LETTER_KEYWORDS = [
        'drag the letter', 'place where it fits'
    ]
    
    SELECT_SMALLER_KEYWORDS = [
        'normally smaller', 'smaller than', 'pick objects that are normally smaller', 'pick objects smaller than', 'pick the objects that are smaller'
    ]
    
    DRAG_FOOD_KEYWORDS = [
        'drag the food', 'feed the hungry animal'
    ]
    
    SELECT_SITTING_KEYWORDS = [
        'thing made for sitting', 'made for sitting', 'pick the thing made for sitting on', 'sitting on'
    ]
    
    SELECT_DRINKING_KEYWORDS = [
        'designed for drinking', 'drinking from', 'identify all objects designed for drinking from'
    ]
    
    SPOT_DIFFERENT_KEYWORDS = [
        'object that is different', 'click on the object that is different'
    ]
    
    SCENE_MATCH_KEYWORDS = [
        'images that fit the scene', 'fit the scene shown', 'fit the scene shown in the example images'
    ]
    
    TOOL_CUT_KEYWORDS = [
        'tool shown in the sample can cut', 'pick objects that the tool shown in the sample can cut'
    ]
    
    IDENTICAL_KEYWORDS = [
        'two elements that are identical', 'elements that are identical'
    ]
    
    # NEW: Fit inside keywords
    FIT_INSIDE_KEYWORDS = [
        'fit inside the sample item', 'fit inside', 'objects that fit inside', 
        'click all the objects that fit inside the sample item'
    ]
    
    # Food and animals
    FOOD_ITEMS = ['turkey', 'chicken', 'meat', 'drumstick']
    ANIMALS = ['bear', 'polar bear']
    
    # Size categories
    SMALL_OBJECTS = ['chess', 'pawn', 'piece', 'yarn', 'ball', 'thread']
    LARGE_OBJECTS = ['pumpkin', 'truck', 'mixer']
    
    # Sitting objects (accepts towel)
    SITTING_OBJECTS = [
        'chair', 'bench', 'stool', 'couch', 'sofa', 'seat', 'towel', 'bean bag'
    ]
    NOT_SITTING = ['toaster', 'truck']
    
    # Drinking objects
    DRINKING_OBJECTS = [
        'cup', 'mug', 'glass', 'tumbler', 'wine glass', 'champagne', 'teacup', 'water glass'
    ]
    NOT_DRINKING = ['avocado', 'pen', 'bowl', 'plate']
    
    # Scene objects
    SCENE_OBJECTS = [
        'apple', 'fruit', 'vegetable', 'produce', 'market', 'grocery'
    ]
    
    # Tool cut objects
    TOOL_CUT_OBJECTS = [
        'thread', 'fabric', 'cloth', 'carrot', 'ribbon', 'sock'
    ]
    
    # NEW: Container items (baskets, boxes, bags, etc.)
    CONTAINER_ITEMS = [
        'basket', 'shopping basket', 'bag', 'box', 'container', 'bucket', 
        'bin', 'cart', 'tote', 'purse', 'backpack', 'suitcase'
    ]
    
    # NEW: Small items that fit in containers
    SMALL_FIT_ITEMS = [
        'cup', 'glass', 'bottle', 'juice', 'orange juice', 'drink', 'beverage', 'soda', 'water',
        'apple', 'orange', 'fruit', 'sandwich', 'food'
    ]
    
    # NEW: Large items that DON'T fit in containers
    LARGE_NO_FIT_ITEMS = [
        'train', 'car', 'truck', 'vehicle', 'bus', 'airplane', 'boat',
        'whale', 'dolphin', 'shark', 'bicycle', 'tire', 'bicycle tire'
    ]
    
    @staticmethod
    def detect_type(prompt: str) -> str:
        """Identify captcha type from prompt text."""
        if not prompt:
            return 'checkbox'
        
        p = prompt.lower()
        
        # Check for NEW fit_inside type FIRST (most specific)
        if any(k in p for k in CaptchaPatterns.FIT_INSIDE_KEYWORDS):
            return 'fit_inside'
        
        if any(k in p for k in CaptchaPatterns.SPOT_DIFFERENT_KEYWORDS):
            return 'spot_different'
        
        if any(k in p for k in CaptchaPatterns.SELECT_SITTING_KEYWORDS):
            return 'select_sitting'
        
        if any(k in p for k in CaptchaPatterns.SELECT_DRINKING_KEYWORDS):
            return 'select_drinking'
        
        if any(k in p for k in CaptchaPatterns.SCENE_MATCH_KEYWORDS):
            return 'scene_match'
        
        if any(k in p for k in CaptchaPatterns.TOOL_CUT_KEYWORDS):
            return 'tool_cut'
        
        if any(k in p for k in CaptchaPatterns.IDENTICAL_KEYWORDS):
            return 'identical'
        
        if any(k in p for k in CaptchaPatterns.DRAG_LETTER_KEYWORDS):
            return 'drag_letter'
        
        if any(k in p for k in CaptchaPatterns.SELECT_SMALLER_KEYWORDS):
            return 'select_smaller'
        
        if any(k in p for k in CaptchaPatterns.DRAG_FOOD_KEYWORDS):
            return 'drag_food'
        
        return 'checkbox'
    
    # --- Category check helpers ---
    
    @staticmethod
    def is_sitting_object(label: str) -> bool:
        if not label:
            return False
        txt = label.lower()
        return any(obj in txt for obj in CaptchaPatterns.SITTING_OBJECTS)
    
    @staticmethod
    def is_drinking_object(label: str) -> bool:
        if not label:
            return False
        txt = label.lower()
        return any(obj in txt for obj in CaptchaPatterns.DRINKING_OBJECTS)
    
    @staticmethod
    def is_scene_match(label: str) -> bool:
        if not label:
            return False
        txt = label.lower()
        return any(obj in txt for obj in CaptchaPatterns.SCENE_OBJECTS)
    
    @staticmethod
    def is_tool_cut(label: str) -> bool:
        if not label:
            return False
        txt = label.lower()
        return any(obj in txt for obj in CaptchaPatterns.TOOL_CUT_OBJECTS)
    
    @staticmethod
    def is_identical(label1: str, label2: str) -> bool:
        return label1.strip().lower() == label2.strip().lower()
    
    # NEW: Fit inside helper
    @staticmethod
    def can_fit_inside(item_label: str, container_label: str = None) -> bool:
        """
        Check if item can physically fit inside a container (basket, box, etc.)
        Returns True if item is small enough to fit inside.
        """
        if not item_label:
            return False
        
        txt = item_label.lower()
        
        # Definitely fits (small items)
        if any(obj in txt for obj in CaptchaPatterns.SMALL_FIT_ITEMS):
            return True
        
        # Definitely doesn't fit (large items)
        if any(obj in txt for obj in CaptchaPatterns.LARGE_NO_FIT_ITEMS):
            return False
        
        # Unknown item - assume doesn't fit (conservative approach)
        return False
    
    @staticmethod
    def is_container(label: str) -> bool:
        """Check if item is a container (basket, box, bag, etc.)"""
        if not label:
            return False
        txt = label.lower()
        return any(obj in txt for obj in CaptchaPatterns.CONTAINER_ITEMS)
    
    # Legacy compatibility helpers
    
    @staticmethod
    def is_food(label: str) -> bool:
        if not label:
            return False
        return any(food in label.lower() for food in CaptchaPatterns.FOOD_ITEMS)
    
    @staticmethod
    def is_animal(label: str) -> bool:
        if not label:
            return False
        return any(animal in label.lower() for food in CaptchaPatterns.ANIMALS)
    
    @staticmethod
    def is_small_object(label: str) -> bool:
        if not label:
            return False
        return any(obj in label.lower() for obj in CaptchaPatterns.SMALL_OBJECTS)
    
    @staticmethod
    def is_large_object(label: str) -> bool:
        if not label:
            return False
        return any(obj in label.lower() for obj in CaptchaPatterns.LARGE_OBJECTS)


# Legacy compatibility
def get_patterns():
    return {}

def match_pattern(image, pattern_ranges):
    return False

class PatternMatcher:
    @staticmethod
    def get_patterns():
        return {}
    
    @staticmethod
    def match_pattern(image, pattern_ranges):
        return False