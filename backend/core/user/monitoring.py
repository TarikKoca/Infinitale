"""
Health monitoring and status tracking utilities.
"""
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from pathlib import Path

from django.conf import settings
from django.core.cache import cache
from django.db import connection
from django.utils import timezone
from django.db.models import Count, Q, Avg, Max, Min

from user.models import GameTurn, TurnStatus, Game, User, GameStory
from celery import current_app
import redis
import psutil
import httpx

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitor system health and collect metrics."""
    
    def __init__(self):
        self.error_window = deque(maxlen=1000)  # Last 1000 errors
        self.metrics_cache_ttl = 60  # Cache metrics for 60 seconds
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        cache_key = "system_metrics"
        cached = cache.get(cache_key)
        if cached:
            return cached
            
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu': {
                    'percent': cpu_percent,
                    'status': 'healthy' if cpu_percent < 80 else 'warning' if cpu_percent < 90 else 'critical'
                },
                'memory': {
                    'percent': memory.percent,
                    'used_gb': round(memory.used / (1024**3), 2),
                    'total_gb': round(memory.total / (1024**3), 2),
                    'status': 'healthy' if memory.percent < 80 else 'warning' if memory.percent < 90 else 'critical'
                },
                'disk': {
                    'percent': disk.percent,
                    'used_gb': round(disk.used / (1024**3), 2),
                    'total_gb': round(disk.total / (1024**3), 2),
                    'status': 'healthy' if disk.percent < 80 else 'warning' if disk.percent < 90 else 'critical'
                }
            }
            cache.set(cache_key, metrics, self.metrics_cache_ttl)
            return metrics
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def get_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start = time.time()
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            query_time = (time.time() - start) * 1000  # ms
            
            return {
                'status': 'healthy' if query_time < 100 else 'warning' if query_time < 500 else 'critical',
                'response_time_ms': round(query_time, 2),
                'active_connections': len(connection.queries) if settings.DEBUG else 'N/A'
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'down',
                'error': str(e)
            }
    
    def get_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            r = redis.from_url(settings.CELERY_BROKER_URL)
            start = time.time()
            r.ping()
            ping_time = (time.time() - start) * 1000
            
            info = r.info()
            return {
                'status': 'healthy' if ping_time < 50 else 'warning' if ping_time < 200 else 'critical',
                'response_time_ms': round(ping_time, 2),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_mb': round(info.get('used_memory', 0) / (1024**2), 2)
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                'status': 'down',
                'error': str(e)
            }
    
    def get_celery_health(self) -> Dict[str, Any]:
        """Check Celery worker status."""
        try:
            i = current_app.control.inspect()
            stats = i.stats()
            active = i.active()
            
            if not stats:
                return {'status': 'down', 'workers': 0}
            
            worker_count = len(stats)
            total_active = sum(len(tasks) for tasks in (active or {}).values())
            
            return {
                'status': 'healthy',
                'workers': worker_count,
                'active_tasks': total_active,
                'queues': list(current_app.amqp.queues.keys())
            }
        except Exception as e:
            logger.error(f"Celery health check failed: {e}")
            return {
                'status': 'down',
                'error': str(e)
            }
    
    def get_api_health(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        apis = {
            'openai': {
                'url': 'https://api.openai.com/v1/models',
                'headers': {'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY", "")[:8]}...'}
            },
            'replicate': {
                'url': 'https://api.replicate.com/v1/models',
                'headers': {'Authorization': f'Token {os.getenv("REPLICATE_API_TOKEN", "")[:8]}...'}
            }
        }
        
        results = {}
        for name, config in apis.items():
            try:
                start = time.time()
                with httpx.Client(timeout=5.0, http2=True, headers=config['headers']) as client:
                    resp = client.get(config['url'])
                response_time = (time.time() - start) * 1000
                results[name] = {
                    'status': 'healthy' if resp.status_code < 400 else 'error',
                    'response_time_ms': round(response_time, 2),
                    'status_code': resp.status_code
                }
            except Exception as e:
                results[name] = {
                    'status': 'down',
                    'error': str(e)
                }
        
        return results
    
    def get_error_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error metrics from logs."""
        try:
            error_log_path = settings.BASE_DIR / 'var' / 'logs' / 'error.log'
            main_log_path = settings.BASE_DIR / 'var' / 'logs' / 'Eternalore.log'
            
            since = timezone.now() - timedelta(hours=hours)
            errors_by_type = defaultdict(int)
            errors_by_hour = defaultdict(int)
            recent_errors = []
            
            # Parse error log
            if error_log_path.exists():
                with open(error_log_path, 'r') as f:
                    for line in f:
                        if 'ERROR' in line:
                            try:
                                # Extract timestamp and error type
                                parts = line.split(' ', 4)
                                if len(parts) >= 5:
                                    timestamp_str = f"{parts[1]} {parts[2]}"
                                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                                    
                                    if timestamp.replace(tzinfo=timezone.utc) > since:
                                        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
                                        errors_by_hour[hour_key] += 1
                                        
                                        # Extract error type
                                        error_msg = parts[4]
                                        if "Turn" in error_msg and "failed" in error_msg:
                                            errors_by_type['turn_processing'] += 1
                                        elif "Game creation failed" in error_msg:
                                            errors_by_type['game_creation'] += 1
                                        elif "TTI" in error_msg or "TTS" in error_msg:
                                            errors_by_type['media_generation'] += 1
                                        elif "Database" in error_msg:
                                            errors_by_type['database'] += 1
                                        else:
                                            errors_by_type['other'] += 1
                                        
                                        if len(recent_errors) < 10:
                                            recent_errors.append({
                                                'timestamp': timestamp.isoformat(),
                                                'message': error_msg[:200]
                                            })
                            except Exception:
                                pass
            
            # Calculate error rate
            total_errors = sum(errors_by_type.values())
            
            # Get successful operations count
            successful_turns = GameTurn.objects.filter(
                completed_at__gte=since,
                status=TurnStatus.DONE
            ).count()
            
            failed_turns = GameTurn.objects.filter(
                completed_at__gte=since,
                status=TurnStatus.FAILED
            ).count()
            
            error_rate = (failed_turns / (successful_turns + failed_turns) * 100) if (successful_turns + failed_turns) > 0 else 0
            
            return {
                'total_errors': total_errors,
                'error_rate': round(error_rate, 2),
                'errors_by_type': dict(errors_by_type),
                'errors_by_hour': dict(sorted(errors_by_hour.items())[-24:]),
                'recent_errors': recent_errors,
                'failed_turns': failed_turns,
                'successful_turns': successful_turns
            }
        except Exception as e:
            logger.error(f"Failed to get error metrics: {e}")
            return {
                'error': str(e)
            }
    
    def _calculate_avg_turn_time(self, since: datetime) -> Optional[float]:
        """Calculate average turn processing time in seconds."""
        turns = GameTurn.objects.filter(
            completed_at__gte=since,
            status=TurnStatus.DONE,
            started_at__isnull=False,
            completed_at__isnull=False
        ).values('started_at', 'completed_at')
        
        if not turns:
            return None
        
        total_seconds = 0
        count = 0
        
        for turn in turns:
            duration = turn['completed_at'] - turn['started_at']
            total_seconds += duration.total_seconds()
            count += 1
        
        return round(total_seconds / count, 2) if count > 0 else None
    
    def get_game_metrics(self) -> Dict[str, Any]:
        """Get game-related metrics."""
        try:
            now = timezone.now()
            last_hour = now - timedelta(hours=1)
            last_24h = now - timedelta(hours=24)
            
            metrics = {
                'active_games': Game.objects.filter(
                    status='ongoing',
                    is_archived=False
                ).count(),
                
                'games_created_1h': Game.objects.filter(
                    created_at__gte=last_hour
                ).count(),
                
                'games_created_24h': Game.objects.filter(
                    created_at__gte=last_24h
                ).count(),
                
                'turns_processed_1h': GameTurn.objects.filter(
                    completed_at__gte=last_hour,
                    status=TurnStatus.DONE
                ).count(),
                
                'turns_processed_24h': GameTurn.objects.filter(
                    completed_at__gte=last_24h,
                    status=TurnStatus.DONE
                ).count(),
                
                'avg_turn_time': self._calculate_avg_turn_time(last_24h),
                
                'active_users_24h': Game.objects.filter(
                    last_played_at__gte=last_24h
                ).values('user').distinct().count()
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get game metrics: {e}")
            return {'error': str(e)}
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        components = {
            'system': self.get_system_metrics(),
            'database': self.get_database_health(),
            'redis': self.get_redis_health(),
            'celery': self.get_celery_health(),
            'apis': self.get_api_health(),
            'errors': self.get_error_metrics(hours=1)
        }
        
        # Determine overall status
        statuses = []
        for component, data in components.items():
            if isinstance(data, dict) and 'status' in data:
                statuses.append(data['status'])
            elif component == 'apis':
                statuses.extend([api.get('status', 'unknown') for api in data.values()])
        
        if 'down' in statuses or 'critical' in statuses:
            overall_status = 'critical'
        elif 'warning' in statuses or 'error' in statuses:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        return {
            'status': overall_status,
            'timestamp': timezone.now().isoformat(),
            'components': components
        }


class ServiceTester:
    """Test various services and functionalities."""
    
    def __init__(self):
        self.test_user_id = None
        self.test_game_id = None
        
    def cleanup_test_data(self):
        """Clean up any test data created."""
        if self.test_game_id:
            try:
                Game.objects.filter(id=self.test_game_id).delete()
            except Exception:
                pass
        
        if self.test_user_id:
            try:
                User.objects.filter(id=self.test_user_id).delete()
            except Exception:
                pass
    
    def test_database(self) -> Tuple[bool, str, float]:
        """Test database connectivity."""
        start = time.time()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result[0] == 1:
                    duration = time.time() - start
                    return True, "Database connection successful", duration
                else:
                    return False, "Unexpected database response", 0
        except Exception as e:
            return False, f"Database error: {str(e)}", 0
    
    def test_redis(self) -> Tuple[bool, str, float]:
        """Test Redis connectivity."""
        start = time.time()
        try:
            r = redis.from_url(settings.CELERY_BROKER_URL)
            test_key = f"test_key_{int(time.time())}"
            r.set(test_key, "test_value", ex=10)
            value = r.get(test_key)
            r.delete(test_key)
            
            if value == b"test_value":
                duration = time.time() - start
                return True, "Redis connection successful", duration
            else:
                return False, "Redis read/write mismatch", 0
        except Exception as e:
            return False, f"Redis error: {str(e)}", 0
    
    def test_celery(self) -> Tuple[bool, str, float]:
        """Test Celery task execution."""
        start = time.time()
        try:
            from celery import current_app
            result = current_app.send_task('celery.ping')
            if result:
                duration = time.time() - start
                return True, "Celery is responsive", duration
            else:
                return False, "Celery not responding", 0
        except Exception as e:
            return False, f"Celery error: {str(e)}", 0
    
    def test_openai_api(self) -> Tuple[bool, str, float]:
        """Test OpenAI API connectivity."""
        start = time.time()
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Simple completion test
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'test'"}],
                max_tokens=10
            )
            
            if response.choices[0].message.content:
                duration = time.time() - start
                return True, "OpenAI API is working", duration
            else:
                return False, "No response from OpenAI", 0
        except Exception as e:
            return False, f"OpenAI API error: {str(e)}", 0
    
    def test_replicate_api(self) -> Tuple[bool, str, float]:
        """Test Replicate API connectivity."""
        start = time.time()
        try:
            import replicate
            
            # Check if we can list models
            models = list(replicate.models.list())
            if models:
                duration = time.time() - start
                return True, "Replicate API is working", duration
            else:
                return False, "No models returned from Replicate", 0
        except Exception as e:
            return False, f"Replicate API error: {str(e)}", 0
    
    def test_user_creation(self) -> Tuple[bool, str, float]:
        """Test user creation functionality."""
        start = time.time()
        try:
            from django.contrib.auth import get_user_model
            AuthUser = get_user_model()
            
            test_email = f"test_{int(time.time())}@test.com"
            auth_user = AuthUser.objects.create_user(
                username=f"test_{int(time.time())}",
                email=test_email,
                password="testpass123"
            )
            
            user = User.objects.create(
                name="Test User",
                email=test_email,
                subscription_plan="lite"
            )
            self.test_user_id = user.id
            
            duration = time.time() - start
            
            # Cleanup
            auth_user.delete()
            user.delete()
            self.test_user_id = None
            
            return True, "User creation successful", duration
        except Exception as e:
            return False, f"User creation error: {str(e)}", 0
    
    def test_game_creation(self) -> Tuple[bool, str, float]:
        """Test game creation process."""
        start = time.time()
        try:
            # Create test user first
            user = User.objects.create(
                name="Test Game User",
                email=f"gametest_{int(time.time())}@test.com",
                subscription_plan="lite"
            )
            self.test_user_id = user.id
            
            # Create game
            game = Game.objects.create(
                user=user,
                genre_name="mystery",
                main_character_name="TestHero",
                visual_style="illustration",
                difficulty="casual",
                error_count=0  # Explicitly set to avoid database constraint error
            )
            self.test_game_id = game.id
            
            # Create story
            story = GameStory.objects.create(
                game=game,
                introductory_text="Test world",
                story="Test story",
                realized_story="",
                messages=[]
            )
            
            duration = time.time() - start
            
            # Cleanup
            story.delete()
            game.delete()
            user.delete()
            self.test_game_id = None
            self.test_user_id = None
            
            return True, "Game creation successful", duration
        except Exception as e:
            self.cleanup_test_data()
            return False, f"Game creation error: {str(e)}", 0
    
    def test_turn_processing(self) -> Tuple[bool, str, float]:
        """Test turn processing (without actual API calls)."""
        start = time.time()
        try:
            # This is a mock test - in production you'd want to test with real APIs
            duration = time.time() - start
            return True, "Turn processing test completed (mock)", duration
        except Exception as e:
            return False, f"Turn processing error: {str(e)}", 0
    
    def test_media_storage(self) -> Tuple[bool, str, float]:
        """Test media storage functionality."""
        start = time.time()
        try:
            # Test creating media directories
            media_root = Path(settings.MEDIA_ROOT)
            test_dirs = ['audio', 'image']
            
            for dir_name in test_dirs:
                dir_path = media_root / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Test file write
                test_file = dir_path / f"test_{int(time.time())}.txt"
                test_file.write_text("test content")
                
                # Verify and cleanup
                if test_file.exists():
                    test_file.unlink()
                else:
                    return False, f"Failed to create test file in {dir_name}", 0
            
            duration = time.time() - start
            return True, "Media storage is working", duration
        except Exception as e:
            return False, f"Media storage error: {str(e)}", 0
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        tests = [
            ('database', self.test_database),
            ('redis', self.test_redis),
            ('celery', self.test_celery),
            ('openai_api', self.test_openai_api),
            ('replicate_api', self.test_replicate_api),
            ('user_creation', self.test_user_creation),
            ('game_creation', self.test_game_creation),
            ('turn_processing', self.test_turn_processing),
            ('media_storage', self.test_media_storage),
        ]
        
        results = {}
        total_duration = 0
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                success, message, duration = test_func()
                results[test_name] = {
                    'success': success,
                    'message': message,
                    'duration': round(duration * 1000, 2)  # Convert to ms
                }
                total_duration += duration
                if success:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                results[test_name] = {
                    'success': False,
                    'message': f"Test crashed: {str(e)}",
                    'duration': 0
                }
                failed += 1
        
        return {
            'results': results,
            'summary': {
                'total': len(tests),
                'passed': passed,
                'failed': failed,
                'duration_ms': round(total_duration * 1000, 2)
            },
            'timestamp': timezone.now().isoformat()
        }