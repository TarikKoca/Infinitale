"""
Comprehensive test suite for monitoring functionality.
"""
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

import django
django.setup()

from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse
from unittest.mock import patch, MagicMock
import json

from user.monitoring import HealthMonitor, ServiceTester
from user.models import User as ProfileUser, Game, GameStory, GameTurn, TurnStatus


class MonitoringTestCase(TestCase):
    """Test monitoring utilities."""
    
    def setUp(self):
        self.monitor = HealthMonitor()
        self.tester = ServiceTester()
    
    def test_system_metrics(self):
        """Test system metrics collection."""
        metrics = self.monitor.get_system_metrics()
        
        self.assertIn('cpu', metrics)
        self.assertIn('memory', metrics)
        self.assertIn('disk', metrics)
        
        # Check CPU metrics
        self.assertIn('percent', metrics['cpu'])
        self.assertIn('status', metrics['cpu'])
        self.assertGreaterEqual(metrics['cpu']['percent'], 0)
        self.assertLessEqual(metrics['cpu']['percent'], 100)
        
        # Check memory metrics
        self.assertIn('percent', metrics['memory'])
        self.assertIn('used_gb', metrics['memory'])
        self.assertIn('total_gb', metrics['memory'])
        self.assertGreater(metrics['memory']['total_gb'], 0)
        
        # Check disk metrics
        self.assertIn('percent', metrics['disk'])
        self.assertIn('used_gb', metrics['disk'])
        self.assertIn('total_gb', metrics['disk'])
        self.assertGreater(metrics['disk']['total_gb'], 0)
    
    def test_database_health(self):
        """Test database health check."""
        health = self.monitor.get_database_health()
        
        self.assertIn('status', health)
        self.assertIn('response_time_ms', health)
        self.assertEqual(health['status'], 'healthy')
        self.assertGreater(health['response_time_ms'], 0)
    
    def test_game_metrics(self):
        """Test game metrics collection."""
        # Create test data
        user = ProfileUser.objects.create(
            name="Test User",
            email="test@example.com",
            subscription_plan="lite"
        )
        
        game = Game.objects.create(
            user=user,
            genre_name="mystery",
            main_character_name="Hero",
            visual_style="illustration",
            difficulty="casual",
            status="ongoing",
            error_count=0  # Explicitly set to avoid database constraint error
        )
        
        metrics = self.monitor.get_game_metrics()
        
        self.assertIn('active_games', metrics)
        self.assertIn('games_created_24h', metrics)
        self.assertIn('turns_processed_24h', metrics)
        self.assertIn('active_users_24h', metrics)
        
        self.assertGreaterEqual(metrics['active_games'], 1)
        self.assertGreaterEqual(metrics['games_created_24h'], 1)
        
        # Cleanup
        game.delete()
        user.delete()
    
    def test_error_metrics(self):
        """Test error metrics collection."""
        metrics = self.monitor.get_error_metrics(hours=24)
        
        self.assertIn('total_errors', metrics)
        self.assertIn('error_rate', metrics)
        self.assertIn('errors_by_type', metrics)
        self.assertIn('errors_by_hour', metrics)
        self.assertIn('recent_errors', metrics)
    
    def test_overall_health(self):
        """Test overall health status."""
        health = self.monitor.get_overall_health()
        
        self.assertIn('status', health)
        self.assertIn('timestamp', health)
        self.assertIn('components', health)
        
        self.assertIn(health['status'], ['healthy', 'warning', 'critical'])
        self.assertIn('system', health['components'])
        self.assertIn('database', health['components'])


class ServiceTesterTestCase(TestCase):
    """Test service testing functionality."""
    
    def setUp(self):
        self.tester = ServiceTester()
    
    def test_database_test(self):
        """Test database connectivity test."""
        success, message, duration = self.tester.test_database()
        
        self.assertTrue(success)
        self.assertEqual(message, "Database connection successful")
        self.assertGreater(duration, 0)
    
    def test_user_creation_test(self):
        """Test user creation functionality test."""
        success, message, duration = self.tester.test_user_creation()
        
        self.assertTrue(success)
        self.assertEqual(message, "User creation successful")
        self.assertGreater(duration, 0)
    
    def test_game_creation_test(self):
        """Test game creation functionality test."""
        success, message, duration = self.tester.test_game_creation()
        
        self.assertTrue(success)
        self.assertEqual(message, "Game creation successful")
        self.assertGreater(duration, 0)
    
    def test_media_storage_test(self):
        """Test media storage functionality test."""
        success, message, duration = self.tester.test_media_storage()
        
        self.assertTrue(success)
        self.assertEqual(message, "Media storage is working")
        self.assertGreater(duration, 0)
    
    def test_run_all_tests(self):
        """Test running all tests."""
        with patch.object(self.tester, 'test_openai_api', return_value=(True, "Mocked", 0.1)):
            with patch.object(self.tester, 'test_replicate_api', return_value=(True, "Mocked", 0.1)):
                with patch.object(self.tester, 'test_redis', return_value=(True, "Mocked", 0.1)):
                    with patch.object(self.tester, 'test_celery', return_value=(True, "Mocked", 0.1)):
                        results = self.tester.run_all_tests()
        
        self.assertIn('results', results)
        self.assertIn('summary', results)
        self.assertIn('timestamp', results)
        
        self.assertGreater(results['summary']['total'], 0)
        self.assertGreaterEqual(results['summary']['passed'], 0)
        self.assertGreaterEqual(results['summary']['failed'], 0)


class AdminViewsTestCase(TestCase):
    """Test admin monitoring views."""
    
    def setUp(self):
        self.client = Client()
        self.User = get_user_model()
        
        # Create staff user
        self.staff_user = self.User.objects.create_user(
            username='staff',
            email='staff@test.com',
            password='staffpass123',
            is_staff=True
        )
        
        # Create regular user
        self.regular_user = self.User.objects.create_user(
            username='regular',
            email='regular@test.com',
            password='regularpass123'
        )
    
    def test_admin_status_page_requires_staff(self):
        """Test that admin status page requires staff permission."""
        # Not logged in
        response = self.client.get(reverse('admin_status'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
        
        # Regular user
        self.client.login(username='regular', password='regularpass123')
        response = self.client.get(reverse('admin_status'))
        self.assertEqual(response.status_code, 302)  # Redirect to login
        
        # Staff user
        self.client.login(username='staff', password='staffpass123')
        response = self.client.get(reverse('admin_status'))
        self.assertEqual(response.status_code, 200)
    
    def test_health_check_api(self):
        """Test health check API endpoint."""
        self.client.login(username='staff', password='staffpass123')
        
        response = self.client.get(reverse('api_health_check'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn('components', data)
    
    def test_system_metrics_api(self):
        """Test system metrics API endpoint."""
        self.client.login(username='staff', password='staffpass123')
        
        response = self.client.get(reverse('api_system_metrics'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('cpu', data)
        self.assertIn('memory', data)
        self.assertIn('disk', data)
    
    def test_error_metrics_api(self):
        """Test error metrics API endpoint."""
        self.client.login(username='staff', password='staffpass123')
        
        response = self.client.get(reverse('api_error_metrics'), {'hours': 24})
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('total_errors', data)
        self.assertIn('error_rate', data)
    
    def test_game_metrics_api(self):
        """Test game metrics API endpoint."""
        self.client.login(username='staff', password='staffpass123')
        
        response = self.client.get(reverse('api_game_metrics'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('active_games', data)
        self.assertIn('games_created_24h', data)
    
    def test_run_test_api(self):
        """Test run single test API endpoint."""
        self.client.login(username='staff', password='staffpass123')
        
        response = self.client.post(
            reverse('api_run_test'),
            json.dumps({'test_name': 'database'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('success', data)
        self.assertIn('message', data)
        self.assertIn('duration_ms', data)
    
    def test_clear_cache_api(self):
        """Test clear cache API endpoint."""
        self.client.login(username='staff', password='staffpass123')
        
        response = self.client.post(reverse('api_clear_cache'))
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('success', data)
        self.assertIn('message', data)
    
    def test_log_tail_api(self):
        """Test log tail API endpoint."""
        self.client.login(username='staff', password='staffpass123')
        
        response = self.client.get(reverse('api_log_tail'), {'type': 'main', 'lines': 10})
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.content)
        self.assertIn('logs', data)
        self.assertIn('count', data)
        self.assertIn('log_type', data)


if __name__ == '__main__':
    import unittest
    unittest.main()