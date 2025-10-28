#!/usr/bin/env python3
"""
===============================================================================
                              SHARP GEN v2025
                  Advanced Discord Token Generator
                                                                            
  Author: Tejv
  Theme: Green & Black with Purple accents
  Features: AI Captcha Solver | Rate Limit Bypass | Email Verification
===============================================================================
"""

import asyncio
import aiohttp
import time
import os
import sys
import random
import base64
import json
import re
import secrets
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# UI and Display
from pystyle import Colorate, Colors, Center, Write, Box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Custom modules
from humanize import generate_username, generate_display_name
from advanced_captcha_solver import AdvancedCaptchaSolver

console = Console()

# Configuration
HCAPTCHA_SITE_KEY = "cs_7ecc4c81c1e7592fffb13154e6a3767956f7902fc0f61c26b7e9c4909a470d71"
HCAPTCHA_SITE_URL = "https://captchapro.preview.emergentagent.com/"

DISCORD_API_VERSION = 10
DISCORD_STABLE_BASE = f"https://discord.com/api/v{DISCORD_API_VERSION}"
DISCORD_CANARY_BASE = f"https://canary.discord.com/api/v{DISCORD_API_VERSION}"


class DiscordRateLimitSemaphore:
    """Advanced semaphore for Discord rate limiting with sliding window"""
    
    def __init__(self, max_requests: int = 50, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times: List[float] = []
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self.lock:
            now = time.time()
            
            # Remove expired timestamps
            self.request_times = [
                t for t in self.request_times 
                if now - t < self.time_window
            ]
            
            # If at capacity, wait until oldest request expires
            if len(self.request_times) >= self.max_requests:
                sleep_time = self.time_window - (now - self.request_times[0]) + 0.01
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    now = time.time()
                    self.request_times = [
                        t for t in self.request_times 
                        if now - t < self.time_window
                    ]
            
            # Record this request
            self.request_times.append(now)
            
    async def __aenter__(self):
        await self.acquire()
        return self
        
    async def __aexit__(self, *args):
        pass


class AdvancedDiscordRateLimiter:
    """Multi-level rate limiter with Canary endpoint rotation"""
    
    def __init__(self):
        self.global_semaphore = DiscordRateLimitSemaphore(50, 1.0)
        self.route_buckets: Dict[str, DiscordRateLimitSemaphore] = {}
        self.bucket_lock = asyncio.Lock()
        self.use_canary = False
        self.request_count = 0
        self.rate_limited_until = 0
        
    def get_base_url(self) -> str:
        """Rotate between Canary and Stable endpoints"""
        self.request_count += 1
        if self.request_count % 5 == 0:
            self.use_canary = not self.use_canary
        return DISCORD_CANARY_BASE if self.use_canary else DISCORD_STABLE_BASE
    
    async def get_route_bucket(self, route: str, method: str) -> DiscordRateLimitSemaphore:
        """Get or create semaphore for specific route"""
        bucket_key = f"{method}:{route}"
        
        async with self.bucket_lock:
            if bucket_key not in self.route_buckets:
                if "register" in route or "auth" in route:
                    self.route_buckets[bucket_key] = DiscordRateLimitSemaphore(1, 6.0)
                elif "verify" in route:
                    self.route_buckets[bucket_key] = DiscordRateLimitSemaphore(2, 5.0)
                elif "users/@me" in route:
                    self.route_buckets[bucket_key] = DiscordRateLimitSemaphore(3, 5.0)
                else:
                    self.route_buckets[bucket_key] = DiscordRateLimitSemaphore(5, 5.0)
                    
            return self.route_buckets[bucket_key]
    
    async def request(
        self, 
        method: str, 
        route: str, 
        proxy: Optional[str] = None,
        **kwargs
    ) -> Optional[aiohttp.ClientResponse]:
        """Make rate-limited request with automatic retry"""
        
        if time.time() < self.rate_limited_until:
            wait_time = self.rate_limited_until - time.time()
            console.print(f"[yellow]Global rate limit active, waiting {wait_time:.1f}s[/yellow]")
            await asyncio.sleep(wait_time)
        
        async with self.global_semaphore:
            route_bucket = await self.get_route_bucket(route, method)
            async with route_bucket:
                base_url = self.get_base_url()
                url = f"{base_url}{route}"
                
                proxy_url = f"http://{proxy}" if proxy else None
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.request(
                            method, 
                            url, 
                            proxy=proxy_url,
                            **kwargs
                        ) as response:
                            if 'X-RateLimit-Remaining' in response.headers:
                                remaining = int(response.headers['X-RateLimit-Remaining'])
                                if remaining < 2:
                                    await asyncio.sleep(0.5)
                            
                            if response.status == 429:
                                retry_after = float(response.headers.get('Retry-After', 2.0))
                                self.rate_limited_until = time.time() + retry_after
                                console.print(f"[red]Rate limited, waiting {retry_after}s[/red]")
                                await asyncio.sleep(retry_after)
                                return await self.request(method, route, proxy=proxy, **kwargs)
                            
                            return response
                            
                except Exception as e:
                    console.print(f"[red]Request error: {e}[/red]")
                    return None


class ProxyManager:
    """Advanced proxy manager with multiple sources and testing"""
    
    def __init__(self):
        self.proxies: List[str] = []
        self.proxy_index = 0
        self.lock = asyncio.Lock()
        self.working_proxies: List[str] = []
        
    async def fetch_from_proxyscrape(self) -> List[str]:
        """Fetch from ProxyScrape API"""
        try:
            console.print("[cyan]Fetching from ProxyScrape...[/cyan]")
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
                    timeout=15
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        proxies = [p.strip() for p in text.split('\n') if p.strip()]
                        console.print(f"[green]ProxyScrape: {len(proxies)} proxies[/green]")
                        return proxies
        except Exception as e:
            console.print(f"[yellow]ProxyScrape failed: {e}[/yellow]")
        return []
    
    async def fetch_from_proxylist_download(self) -> List[str]:
        """Fetch from proxy-list.download"""
        try:
            console.print("[cyan]Fetching from proxy-list.download...[/cyan]")
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.proxy-list.download/api/v1/get?type=http",
                    timeout=15
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        proxies = [p.strip() for p in text.split('\n') if p.strip()]
                        console.print(f"[green]proxy-list.download: {len(proxies)} proxies[/green]")
                        return proxies
        except Exception as e:
            console.print(f"[yellow]proxy-list.download failed: {e}[/yellow]")
        return []
    
    async def fetch_from_pubproxy(self) -> List[str]:
        """Fetch from PubProxy API"""
        try:
            console.print("[cyan]Fetching from PubProxy...[/cyan]")
            proxies = []
            async with aiohttp.ClientSession() as session:
                for _ in range(5):
                    async with session.get(
                        "http://pubproxy.com/api/proxy?limit=20&format=txt&type=http",
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            text = await response.text()
                            batch = [p.strip() for p in text.split('\n') if p.strip()]
                            proxies.extend(batch)
                    await asyncio.sleep(0.5)
            
            console.print(f"[green]PubProxy: {len(proxies)} proxies[/green]")
            return proxies
        except Exception as e:
            console.print(f"[yellow]PubProxy failed: {e}[/yellow]")
        return []
    
    async def fetch_from_freeproxylist(self) -> List[str]:
        """Fetch from free-proxy-list.net"""
        try:
            console.print("[cyan]Fetching from free-proxy-list.net...[/cyan]")
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://free-proxy-list.net/",
                    timeout=15
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})</td><td>(\d+)</td>'
                        matches = re.findall(pattern, text)
                        proxies = [f"{ip}:{port}" for ip, port in matches]
                        console.print(f"[green]free-proxy-list.net: {len(proxies)} proxies[/green]")
                        return proxies[:100]
        except Exception as e:
            console.print(f"[yellow]free-proxy-list.net failed: {e}[/yellow]")
        return []
    
    async def test_proxy(self, proxy: str) -> bool:
        """Test if a proxy is working"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://httpbin.org/ip",
                    proxy=f"http://{proxy}",
                    timeout=8
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def test_proxies(self, proxies: List[str]) -> List[str]:
        """Test multiple proxies concurrently"""
        console.print(f"[cyan]Testing {len(proxies)} proxies...[/cyan]")
        
        working = []
        tasks = []
        
        for proxy in proxies[:200]:
            tasks.append(self.test_proxy(proxy))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for proxy, is_working in zip(proxies[:200], results):
            if is_working is True:
                working.append(proxy)
                if len(working) >= 20:
                    break
        
        console.print(f"[green]Found {len(working)} working proxies[/green]")
        return working
    
    def load_from_file(self) -> List[str]:
        """Load proxies from proxies.txt"""
        if os.path.exists('proxies.txt'):
            with open('proxies.txt', 'r') as f:
                proxies = [line.strip() for line in f if line.strip()]
                if proxies:
                    console.print(f"[green]Loaded {len(proxies)} proxies from file[/green]")
                    return proxies
        return []
    
    async def initialize(self):
        """Initialize proxy pool from all sources"""
        console.print("\n[bold cyan]Initializing Proxy Manager[/bold cyan]")
        
        file_proxies = self.load_from_file()
        if file_proxies:
            tested = await self.test_proxies(file_proxies)
            if tested:
                self.proxies = tested
                self.working_proxies = tested
                console.print(f"[green]Using {len(tested)} proxies from file[/green]\n")
                return
        
        tasks = [
            self.fetch_from_proxyscrape(),
            self.fetch_from_proxylist_download(),
            self.fetch_from_pubproxy(),
            self.fetch_from_freeproxylist()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_proxies = []
        for result in results:
            if isinstance(result, list):
                all_proxies.extend(result)
        
        all_proxies = list(set(all_proxies))
        console.print(f"[cyan]Total proxies collected: {len(all_proxies)}[/cyan]")
        
        if all_proxies:
            working = await self.test_proxies(all_proxies)
            if working:
                self.proxies = working
                self.working_proxies = working
                console.print(f"[bold green]Proxy initialization complete: {len(working)} working proxies[/bold green]\n")
                return
        
        console.print("[yellow]No working proxies found, running without proxy support[/yellow]\n")
    
    async def get_proxy(self) -> Optional[str]:
        """Get next proxy from rotation"""
        async with self.lock:
            if not self.proxies:
                return None
            
            proxy = self.proxies[self.proxy_index]
            self.proxy_index = (self.proxy_index + 1) % len(self.proxies)
            return proxy


class TempMailProvider:
    """Multi-provider temporary email service"""
    
    @staticmethod
    async def gen_mailtm() -> Tuple[str, str, Dict]:
        """Generate email using mail.tm"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.mail.tm/domains", timeout=10) as resp:
                    domains = await resp.json()
                    domain = domains[0]['domain']
                
                username = secrets.token_hex(6)
                email = f"{username}@{domain}"
                password = secrets.token_hex(12)
                
                async with session.post(
                    "https://api.mail.tm/accounts",
                    json={"address": email, "password": password},
                    timeout=10
                ) as resp:
                    if resp.status == 201:
                        async with session.post(
                            "https://api.mail.tm/token",
                            json={"address": email, "password": password},
                            timeout=10
                        ) as token_resp:
                            data = await token_resp.json()
                            return email, data['token'], {'provider': 'mailtm', 'password': password}
        except:
            pass
        return None, None, {}
    
    @staticmethod
    def gen_1secmail() -> Tuple[str, str, Dict]:
        """Generate email using 1secmail"""
        username = secrets.token_hex(6)
        email = f"{username}@1secmail.com"
        return email, username, {'provider': '1secmail'}
    
    @staticmethod
    def gen_tempmail_lol() -> Tuple[str, str, Dict]:
        """Generate email using tempmail.lol"""
        username = secrets.token_hex(6)
        email = f"{username}@tempmail.lol"
        return email, username, {'provider': 'tempmail_lol'}
    
    @staticmethod
    async def generate() -> Tuple[str, str, Dict]:
        """Generate temp email from available providers"""
        email, token, context = await TempMailProvider.gen_mailtm()
        if email:
            return email, token, context
        
        return TempMailProvider.gen_1secmail()
    
    @staticmethod
    async def poll_verification_code(
        email: str, 
        token: str, 
        context: Dict, 
        timeout: int = 120
    ) -> Optional[str]:
        """Poll for verification code based on provider"""
        provider = context.get('provider', '1secmail')
        
        if provider == 'mailtm':
            return await TempMailProvider._poll_mailtm(email, token, timeout)
        elif provider == '1secmail':
            return await TempMailProvider._poll_1secmail(token, timeout)
        elif provider == 'tempmail_lol':
            return await TempMailProvider._poll_1secmail(token, timeout)
        
        return None
    
    @staticmethod
    async def _poll_mailtm(email: str, token: str, timeout: int) -> Optional[str]:
        """Poll mail.tm for verification code"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {token}"}
                    async with session.get(
                        "https://api.mail.tm/messages",
                        headers=headers,
                        timeout=5
                    ) as response:
                        if response.status == 200:
                            messages = await response.json()
                            for msg in messages:
                                async with session.get(
                                    f"https://api.mail.tm/messages/{msg['id']}",
                                    headers=headers,
                                    timeout=5
                                ) as msg_resp:
                                    data = await msg_resp.json()
                                    text = data.get('text', '') + data.get('html', '')
                                    match = re.search(r'(\d{6})', text)
                                    if match:
                                        return match.group(1)
            except:
                pass
            await asyncio.sleep(4)
        
        return None
    
    @staticmethod
    async def _poll_1secmail(login: str, timeout: int) -> Optional[str]:
        """Poll 1secmail for verification code"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://www.1secmail.com/api/v1/?action=getMessages&login={login}&domain=1secmail.com",
                        timeout=5
                    ) as response:
                        messages = await response.json()
                        
                        for msg in messages:
                            msg_id = msg['id']
                            async with session.get(
                                f"https://www.1secmail.com/api/v1/?action=readMessage&login={login}&domain=1secmail.com&id={msg_id}",
                                timeout=5
                            ) as body_resp:
                                body_data = await body_resp.json()
                                body = body_data.get('body', '')
                                match = re.search(r'(\d{6})', body)
                                if match:
                                    return match.group(1)
            except:
                pass
            await asyncio.sleep(4)
        
        return None


class SharpGen:
    """Main SharpGen token generator class"""
    
    def __init__(self):
        self.rate_limiter = AdvancedDiscordRateLimiter()
        self.proxy_manager = ProxyManager()
        self.captcha_solver = None
        self.used_captcha_keys = set()
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'ev_tokens': 0
        }
        
    def sharpgenn(self):
        """Display Ascii"""
        sharpgen_ = """
███████╗██╗  ██╗ █████╗ ██████╗ ██████╗      ██████╗ ███████╗███╗   ██╗
██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗    ██╔════╝ ██╔════╝████╗  ██║
███████╗███████║███████║██████╔╝██████╔╝    ██║  ███╗█████╗  ██╔██╗ ██║
╚════██║██╔══██║██╔══██║██╔══██╗██╔═══╝     ██║   ██║██╔══╝  ██║╚██╗██║
███████║██║  ██║██║  ██║██║  ██║██║         ╚██████╔╝███████╗██║ ╚████║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝          ╚═════╝ ╚══════╝╚═╝  ╚═══╝
        """
        os.system('cls' if os.name == 'nt' else 'clear')
        print(Colorate.Vertical(Colors.green_to_black, Center.XCenter(sharpgen_)))
        
        info = Panel(
            "[bold green]SharpGen[/bold green] [purple]v2025[/purple] - [cyan]Advanced Discord Token Generator[/cyan]\n"
            "[dim]Author: Tejv[/dim]\n"
            "[yellow]Features:[/yellow] AI Captcha | Rate Limit Bypass | Email Verification | Proxy Rotation",
            border_style="green",
            title="[bold purple]SHARP GEN[/bold purple]",
            title_align="center"
        )
        console.print(info, justify="center")
        console.print()
    
    def _build_headers(self) -> Dict[str, str]:
        """Build Discord API headers"""
        browser_ver = "121.0.0.0"
        super_props = {
            "os": "Windows",
            "browser": "Chrome",
            "device": "",
            "browser_user_agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser_ver} Safari/537.36",
            "browser_version": browser_ver,
            "os_version": "10",
            "referrer": "https://discord.com",
            "referring_domain": "discord.com",
            "release_channel": "stable",
            "client_build_number": 245662
        }
        
        return {
            "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{browser_ver} Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "Origin": "https://discord.com",
            "Referer": "https://discord.com/register",
            "X-Super-Properties": base64.b64encode(json.dumps(super_props).encode()).decode()
        }
    
    async def get_fingerprint(self, proxy: Optional[str] = None) -> str:
        """Get Discord fingerprint"""
        try:
            response = await self.rate_limiter.request(
                "GET", 
                "/experiments", 
                proxy=proxy,
                headers=self._build_headers()
            )
            if response and response.status == 200:
                data = await response.json()
                return data.get("fingerprint", secrets.token_hex(9))
        except:
            pass
        return secrets.token_hex(9)
    
    async def solve_captcha(self) -> Optional[str]:
        """Solve captcha using advanced solver"""
        try:
            if not self.captcha_solver:
                self.captcha_solver = AdvancedCaptchaSolver()
            
            console.print("[cyan]Solving captcha...[/cyan]")
            token = await asyncio.to_thread(
                self.captcha_solver.solve_captcha,
                HCAPTCHA_SITE_KEY,
                HCAPTCHA_SITE_URL
            )
            
            if token and token not in self.used_captcha_keys:
                self.used_captcha_keys.add(token)
                console.print("[green]Captcha solved successfully[/green]")
                return token
            
            console.print("[yellow]Captcha token reused, retrying...[/yellow]")
            return await self.solve_captcha()
            
        except Exception as e:
            console.print(f"[red]Captcha error: {e}[/red]")
            return None
    
    async def fetch_avatar_b64(self) -> Optional[str]:
        """Get random avatar"""
        try:
            seed = secrets.token_hex(8)
            url = f"https://api.dicebear.com/7.x/adventurer/png?seed={seed}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.read()
                        return "data:image/png;base64," + base64.b64encode(content).decode()
        except:
            pass
        return None
    
    async def verify_email(self, token: str, code: str, proxy: Optional[str] = None) -> bool:
        """Verify Discord email"""
        try:
            headers = self._build_headers()
            headers["Authorization"] = token
            
            response = await self.rate_limiter.request(
                "POST",
                "/auth/verify",
                proxy=proxy,
                headers=headers,
                json={"code": code}
            )
            
            return response and response.status == 200
        except:
            return False
    
    async def set_display_name(self, token: str, display_name: str, proxy: Optional[str] = None) -> bool:
        """Set display name"""
        try:
            headers = self._build_headers()
            headers["Authorization"] = token
            
            response = await self.rate_limiter.request(
                "PATCH",
                "/users/@me",
                proxy=proxy,
                headers=headers,
                json={"global_name": display_name}
            )
            
            return response and response.status == 200
        except:
            return False
    
    async def set_avatar(self, token: str, proxy: Optional[str] = None) -> bool:
        """Set random avatar"""
        try:
            avatar_b64 = await self.fetch_avatar_b64()
            if not avatar_b64:
                return False
            
            headers = self._build_headers()
            headers["Authorization"] = token
            
            response = await self.rate_limiter.request(
                "PATCH",
                "/users/@me",
                proxy=proxy,
                headers=headers,
                json={"avatar": avatar_b64}
            )
            
            return response and response.status == 200
        except:
            return False
    
    async def register_account(
        self,
        username: str,
        email: str,
        password: str,
        fingerprint: str,
        captcha_key: str,
        proxy: Optional[str] = None
    ) -> Optional[str]:
        """Register Discord account"""
        dob = f"19{random.randint(85, 99)}-0{random.randint(1,9)}-{random.randint(10,28)}"
        
        payload = {
            "fingerprint": fingerprint,
            "username": username,
            "email": email,
            "password": password,
            "consent": True,
            "date_of_birth": dob,
            "captcha_key": captcha_key
        }
        
        try:
            response = await self.rate_limiter.request(
                "POST",
                "/auth/register",
                proxy=proxy,
                headers=self._build_headers(),
                json=payload
            )
            
            if response and response.status == 201:
                data = await response.json()
                return data.get("token")
            elif response:
                error_text = await response.text()
                console.print(f"[red]Registration failed:[/red] {error_text[:100]}")
        except Exception as e:
            console.print(f"[red]Registration error: {e}[/red]")
        
        return None
    
    async def normal_gen(self) -> Optional[str]:
        """Generate normal token"""
        start_time = time.time()
        self.stats['total'] += 1
        
        console.print("\n[bold cyan]Normal Token Generation[/bold cyan]")
        
        proxy = await self.proxy_manager.get_proxy()
        if proxy:
            console.print(f"[dim]Using proxy: {proxy[:20]}...[/dim]")
        
        username = next(username = generate_username())
        email = f"{username}@tempmail.lol"
        password = f"Pass{secrets.token_hex(6)}"
        
        console.print(f"[dim]Username: {username}[/dim]")
        
        fingerprint = await self.get_fingerprint(proxy)
        console.print("[green]Fingerprint obtained[/green]")
        
        captcha_key = await self.solve_captcha()
        if not captcha_key:
            self.stats['failed'] += 1
            return None
        
        console.print("[cyan]Registering account...[/cyan]")
        token = await self.register_account(username, email, password, fingerprint, captcha_key, proxy)
        
        if token:
            elapsed = time.time() - start_time
            self.stats['success'] += 1
            
            with open("tokens.txt", "a") as f:
                f.write(f"{token}\n")
            
            token_display = Colorate.Horizontal(Colors.black_to_green, token)
            console.print(f"\n[bold green]TOKEN GENERATED[/bold green]")
            console.print(f"   {token_display}")
            console.print(f"   [dim green][ {elapsed:.1f} sec ][/dim green]\n")
            
            return token
        else:
            self.stats['failed'] += 1
            console.print("[red]Token generation failed[/red]")
            return None
    
    async def ev_gen(self) -> Optional[str]:
        """Generate Email-Verified token with PFP and display name"""
        start_time = time.time()
        self.stats['total'] += 1
        
        console.print("\n[bold purple]Email-Verified Token Generation[/bold purple]")
        
        proxy = await self.proxy_manager.get_proxy()
        if proxy:
            console.print(f"[dim]Using proxy: {proxy[:20]}...[/dim]")
        
        username = next(username = generate_username())
        display_name = generate_display_name()
        email, mail_token, mail_context = await TempMailProvider.generate()
        password = f"Pass{secrets.token_hex(6)}"
        
        console.print(f"[dim]Username: {username}[/dim]")
        console.print(f"[dim]Display: {display_name}[/dim]")
        console.print(f"[dim]Email: {email}[/dim]")
        
        fingerprint = await self.get_fingerprint(proxy)
        console.print("[green]Fingerprint obtained[/green]")
        
        captcha_key = await self.solve_captcha()
        if not captcha_key:
            self.stats['failed'] += 1
            return None
        
        console.print("[cyan]Registering account...[/cyan]")
        token = await self.register_account(username, email, password, fingerprint, captcha_key, proxy)
        
        if not token:
            self.stats['failed'] += 1
            console.print("[red]Registration failed[/red]")
            return None
        
        console.print("[green]Account registered[/green]")
        
        console.print("[cyan]Waiting for verification email...[/cyan]")
        code = await TempMailProvider.poll_verification_code(email, mail_token, mail_context, timeout=120)
        
        if not code:
            console.print("[yellow]Verification timeout[/yellow]")
            self.stats['failed'] += 1
            return None
        
        console.print(f"[green]Code received: {code}[/green]")
        
        console.print("[cyan]Verifying email...[/cyan]")
        if not await self.verify_email(token, code, proxy):
            console.print("[red]Email verification failed[/red]")
            self.stats['failed'] += 1
            return None
        
        console.print("[green]Email verified[/green]")
        
        console.print("[cyan]Setting profile...[/cyan]")
        await self.set_display_name(token, display_name, proxy)
        await self.set_avatar(token, proxy)
        console.print("[green]Profile configured[/green]")
        
        elapsed = time.time() - start_time
        self.stats['success'] += 1
        self.stats['ev_tokens'] += 1
        
        with open("ev_tokens.txt", "a") as f:
            f.write(f"{token}\n")
        
        token_display = Colorate.Horizontal(Colors.black_to_green, token)
        console.print(f"\n[bold purple]EV TOKEN GENERATED[/bold purple]")
        console.print(f"   {token_display}")
        console.print(f"   [dim green][ {elapsed:.1f} sec ][ Verified ][ PFP Set ][/dim green]\n")
        
        return token
    
    def display_stats(self):
        """Display generation statistics"""
        table = Table(title="[bold green]Generation Statistics[/bold green]", border_style="green")
        
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Total Attempts", str(self.stats['total']))
        table.add_row("Successful", f"[green]{self.stats['success']}[/green]")
        table.add_row("Failed", f"[red]{self.stats['failed']}[/red]")
        table.add_row("EV Tokens", f"[purple]{self.stats['ev_tokens']}[/purple]")
        
        if self.stats['total'] > 0:
            success_rate = (self.stats['success'] / self.stats['total']) * 100
            table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        console.print(table)
    
    async def menu(self):
        """Main menu loop"""
        self.sharpgenn()
        
        await self.proxy_manager.initialize()
        
        while True:
            menu_text = """
[bold green]SHARP GEN MENU[/bold green]

[purple]1.[/purple] [cyan]Normal Gen[/cyan]        - Standard token generation
[purple]2.[/purple] [green]EV Gen[/green]            - Email verified + PFP + Display
[purple]3.[/purple] [yellow]Batch Normal[/yellow]      - Generate multiple normal tokens
[purple]4.[/purple] [magenta]Batch EV[/magenta]          - Generate multiple EV tokens
[purple]5.[/purple] [blue]Statistics[/blue]        - View generation stats
[purple]0.[/purple] [red]Exit[/red]              - Close SharpGen
            """
            console.print(menu_text)
            
            try:
                Write.Print(">>> Choose option: ", Colors.green_to_black, end="")
                choice = input().strip()
                
                if choice == '1':
                    await self.normal_gen()
                    
                elif choice == '2':
                    await self.ev_gen()
                    
                elif choice == '3':
                    count_str = input(Colorate.Horizontal(Colors.green_to_black, "Enter count: "))
                    count = int(count_str)
                    
                    for i in range(count):
                        console.print(f"\n[bold cyan]Generating {i+1}/{count}[/bold cyan]")
                        await self.normal_gen()
                        if i < count - 1:
                            await asyncio.sleep(random.uniform(2, 5))
                    
                    self.display_stats()
                    
                elif choice == '4':
                    count_str = input(Colorate.Horizontal(Colors.green_to_black, "Enter count: "))
                    count = int(count_str)
                    
                    for i in range(count):
                        console.print(f"\n[bold purple]Generating EV {i+1}/{count}[/bold purple]")
                        await self.ev_gen()
                        if i < count - 1:
                            await asyncio.sleep(random.uniform(3, 6))
                    
                    self.display_stats()
                    
                elif choice == '5':
                    self.display_stats()
                    
                elif choice == '0':
                    console.print(Colorate.Horizontal(
                        Colors.green_to_black,
                        "\nThanks for using SharpGen by Tejv\n"
                    ))
                    if self.captcha_solver:
                        self.captcha_solver.close()
                    break
                    
                else:
                    console.print("[red]Invalid choice[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                if self.captcha_solver:
                    self.captcha_solver.close()
                break
            except ValueError:
                console.print("[red]Invalid input[/red]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


async def main():
    """Main entry point"""
    gen = SharpGen()
    await gen.menu()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
    except Exception as e:

        console.print(f"[red]Fatal error: {e}[/red]")
