# humanize.py
# Account Humanization Module for SharpGen
# Author: Tejv

import random
import string
import secrets
import base64
import aiohttp
from typing import Optional, Dict, Tuple

# ============================================================================
# USERNAME GENERATION
# ============================================================================

# Unique first name components
first_names = [
    'k4iiz', 'za1r11a', 'l100i', 'nk10o', '3zjra', 'arlo', 'm82o', 'f1zznn', 'r011o', 'ax3lzz8',
    'z16in', '0ri0nnzn', 'kn0zzox', 's3ge', '1tlas', 'fe3lzx', 'j3d3ze', 'enn3z0o', 'r4vvi1', 'd2nt33e',
    'lun4r', 'v0rt3x', 'n3bul4', 'qu4s4r', 'puls4r', 'v3ct0r', 'z3ph7r', 'chr0m4', 'xen0n', 'ky0t0',
    'az4r3', 'nyx0s', 'ph4nt0m', 'gl1tch', 'v4p0r', 'r4v3n', 'sh4d0w', 'c0sm0s', 'syr4x', 'thr4x'
]

# Unique last name components
last_names = [
    'cr0oszzs', 'st0nze', 'wi1d3ze', 'fr0z8t', 'ph03n1zzx', 'h1kzwk', 'st0rmm', 'kn1ghhtz', 's1nghh', 'st33ell',
    '5m00nnz', 'g2l1d3err', '3mb3rrz', 'theg43yyz', 'grwats1g3z', 'sprn0vaaz', 'r1v33rz', 'cl3v3rrf0xz', 'w00llf', '1rch3rzr',
    'bl4z3', 'sty7x', 'chr0n1c', 'v0rt1c3', 'qu4nt4', 'sph1nx', 'z3n1th', 'd3lt4', '0m3g4', 's1gm4',
    'puls3', 'n3x4s', 'hy7p3r', 'm1r4g3', 't3mp3st', 'fr4ct4l', 'pr1sm', 's7ry7x', 'c4dz3nc3', 'r4d1x'
]

# Display name components (more normal looking)
display_first = [
    'Kai', 'Zara', 'Leo', 'Niko', 'Ezra', 'Arlo', 'Milo', 'Finn', 'Rio', 'Axel',
    'Luna', 'Nova', 'Ash', 'Sky', 'Rex', 'Max', 'Jade', 'Cole', 'Sage', 'Blake',
    'Kira', 'Zane', 'Aria', 'Jax', 'Echo', 'Raven', 'Vex', 'Onyx', 'Cruz', 'Koda',
    'Riley', 'Quinn', 'Reese', 'Phoenix', 'Storm', 'Winter', 'Ember', 'Atlas', 'Indigo', 'River'
]

display_last = [
    'Cross', 'Stone', 'Wilde', 'Frost', 'Phoenix', 'Hawk', 'Storm', 'Knight', 'Ash', 'Steel',
    'Moon', 'Ocean', 'Ember', 'Grey', 'Nova', 'River', 'Fox', 'Wolf', 'Archer', 'Blade',
    'Shadow', 'Viper', 'Ryder', 'Hunter', 'Blaze', 'Sage', 'Cipher', 'Reign', 'Atlas', 'Drift',
    'Vale', 'Frost', 'Wilder', 'Chase', 'Reed', 'Steele', 'Brooks', 'Wolfe', 'North', 'West'
]


def generate_username() -> str:
    """Generate unique Discord username with optional middle characters"""
    letters_digits = string.ascii_lowercase + string.digits
    middle_chars_list = []
    
    # Generate middle character combinations
    for length in [3, 4]:
        for _ in range(5):
            combo = ''.join(random.choices(letters_digits, k=length))
            middle_chars_list.append(combo)
    
    candidates = [''] + middle_chars_list
    
    first = random.choice(first_names)
    last = random.choice(last_names)
    
    for middle in candidates:
        username = f"{first}{middle}{last}"
        if len(username) <= 32:
            return username
    
    # Fallback
    return f"{first}{last}"[:32]


def generate_display_name() -> str:
    """Generate realistic display name for Discord"""
    first = random.choice(display_first)
    last = random.choice(display_last)
    
    # Variations
    variants = [
        f"{first} {last}",
        f"{first}{last}",
        f"{first} {last[0]}.",
        f"{first} {last} {random.randint(1, 99)}",
        first,
        f"{first}{random.randint(1, 999)}"
    ]
    
    return random.choice(variants)


# ============================================================================
# BIO GENERATION
# ============================================================================

bio_templates = [
    "just vibing",
    "living my best life",
    "chaos coordinator",
    "professional overthinker",
    "caffeine dependent",
    "sleep deprived",
    "daydreamer",
    "night owl",
    "music enthusiast",
    "gamer at heart",
    "anime lover",
    "artist in progress",
    "coding wizard",
    "meme connoisseur",
    "cat person",
    "dog person",
    "pizza lover",
    "coffee addict",
    "tea enthusiast",
    "bookworm",
    "fitness junkie",
    "travel addict",
    "food critic",
    "movie buff",
    "series binger",
    "pro procrastinator",
    "certified overthinker",
    "sleep is for the weak",
    "still loading...",
    "get better",
    "https://dsc.gg/tejv",
    "error 404: bio not found",
    "powered by caffeine",
    "probably sleeping",
    "definitely overthinking",
    "chronically online",
    "touch grass? never heard of it",
    "main character energy",
    "side character vibes",
    "NPC mode activated",
]

bio_emojis = [
    "", "✨", "🌙", "☁️", "🌊", "🔥", "⚡", "🌟", 
    "💫", "🎮", "🎵", "🎨", "📚", "☕", "🍕", "🌸",
    "🦋", "🐱", "🐶", "🎭", "🎪", "🎨", "🎬", "📷"
]


def generate_bio() -> str:
    """Generate a random Discord bio"""
    template = random.choice(bio_templates)
    
    # 40% chance to add emoji
    if random.random() < 0.4:
        emoji = random.choice(bio_emojis)
        if random.random() < 0.5:
            return f"{emoji} {template}"
        else:
            return f"{template} {emoji}"
    
    return template


# ============================================================================
# PRONOUNS
# ============================================================================

pronoun_sets = [
    "he/him",
    "she/her",
    "they/them",
    "he/they",
    "she/they",
    "any/all",
    "bat/man",
    "tej/goat",
    "better",
]


def generate_pronouns() -> str:
    """Generate random pronouns"""
    return random.choice(pronoun_sets)


# ============================================================================
# AVATAR GENERATION
# ============================================================================

async def fetch_avatar_b64(style: str = "adventurer") -> Optional[str]:
    """
    Fetch random avatar from DiceBear API
    
    Available styles:
    - adventurer (neutral, cartoon)
    - avataaars (pixar-style)
    - bottts (robot)
    - croodles (doodle)
    - lorelei (simple illustrations)
    - notionists (notion-style)
    - personas (professional)
    - pixel-art (8-bit style)
    """
    try:
        seed = secrets.token_hex(8)
        url = f"https://api.dicebear.com/7.x/{style}/png?seed={seed}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.read()
                    return "data:image/png;base64," + base64.b64encode(content).decode()
    except Exception as e:
        print(f"Avatar fetch error: {e}")
    
    return None


# ============================================================================
# PROFILE HUMANIZATION
# ============================================================================

class ProfileHumanizer:
    """Main class for humanizing Discord profiles"""
    
    @staticmethod
    def generate_complete_profile() -> Dict[str, str]:
        """Generate complete profile data for EV Gen"""
        return {
            'username': generate_username(),
            'display_name': generate_display_name(),
            'bio': generate_bio(),
            'pronouns': generate_pronouns() if random.random() < 0.6 else None,  # 60% chance
        }
    
    @staticmethod
    def generate_simple_profile() -> Dict[str, str]:
        """Generate simple profile data for Token Gen (no extras)"""
        return {
            'username': generate_username(),
        }
    
    @staticmethod
    async def apply_profile_updates(
        token: str,
        rate_limiter,
        proxy: Optional[str] = None,
        profile_data: Optional[Dict] = None,
        include_avatar: bool = True
    ) -> Dict[str, bool]:
        """
        Apply profile updates to Discord account
        
        Returns dict with success status for each update
        """
        results = {}
        headers = {
            "Authorization": token,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        
        if not profile_data:
            return results
        
        # Update display name (global_name)
        if 'display_name' in profile_data and profile_data['display_name']:
            try:
                response = await rate_limiter.request(
                    "PATCH",
                    "/users/@me",
                    proxy=proxy,
                    headers=headers,
                    json={"global_name": profile_data['display_name']}
                )
                results['display_name'] = response and response.status == 200
            except:
                results['display_name'] = False
        
        # Update bio
        if 'bio' in profile_data and profile_data['bio']:
            try:
                response = await rate_limiter.request(
                    "PATCH",
                    "/users/@me",
                    proxy=proxy,
                    headers=headers,
                    json={"bio": profile_data['bio']}
                )
                results['bio'] = response and response.status == 200
            except:
                results['bio'] = False
        
        # Update pronouns
        if 'pronouns' in profile_data and profile_data['pronouns']:
            try:
                response = await rate_limiter.request(
                    "PATCH",
                    "/users/@me",
                    proxy=proxy,
                    headers=headers,
                    json={"pronouns": profile_data['pronouns']}
                )
                results['pronouns'] = response and response.status == 200
            except:
                results['pronouns'] = False
        
        # Update avatar
        if include_avatar:
            try:
                avatar_styles = ['adventurer', 'avataaars', 'lorelei', 'notionists']
                style = random.choice(avatar_styles)
                avatar_b64 = await fetch_avatar_b64(style)
                
                if avatar_b64:
                    response = await rate_limiter.request(
                        "PATCH",
                        "/users/@me",
                        proxy=proxy,
                        headers=headers,
                        json={"avatar": avatar_b64}
                    )
                    results['avatar'] = response and response.status == 200
                else:
                    results['avatar'] = False
            except:
                results['avatar'] = False
        
        return results
    
    @staticmethod
    def get_humanization_summary(results: Dict[str, bool]) -> str:
        """Get a summary string of humanization results"""
        updates = []
        if results.get('display_name'):
            updates.append("Display Name")
        if results.get('bio'):
            updates.append("Bio")
        if results.get('pronouns'):
            updates.append("Pronouns")
        if results.get('avatar'):
            updates.append("Avatar")
        
        if updates:
            return f"{', '.join(updates)} set"
        else:
            return "No updates applied"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def handle_username_taken_error(current_username: str) -> str:
    """Generate alternative username if current one is taken"""
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(2, 4)))
    base = current_username.rstrip(string.ascii_lowercase + string.digits + '._x')
    new_username = f"{base}{suffix}"
    return new_username[:32]


def generate_password(length: int = 16) -> str:
    """Generate secure random password"""
    return f"Pass{secrets.token_hex(length // 2)}"


def generate_email(username: str) -> str:
    """Generate email address for account"""
    domains = ['tempmail.lol', 'gmail.com', 'outlook.com', 'proton.me']
    return f"{username}@{random.choice(domains)}"