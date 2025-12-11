"""
Management command to clean up stuck game creation processes.

This command identifies games that have been in 'creating' status for more than
10 minutes and cleans them up by removing Redis locks and marking them as failed.
"""

import redis
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone
from user.models import Game
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Clean up stuck game creation processes'

    def add_arguments(self, parser):
        parser.add_argument(
            '--timeout',
            type=int,
            default=600,
            help='Timeout in seconds (default: 600 = 10 minutes)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be cleaned up without actually doing it'
        )

    def handle(self, *args, **options):
        timeout = options['timeout']
        dry_run = options['dry_run']
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No changes will be made')
            )
        
        # Find games stuck in creating status
        cutoff_time = timezone.now() - timezone.timedelta(seconds=timeout)
        stuck_games = Game.objects.filter(
            status='creating',
            created_at__lt=cutoff_time
        )
        
        if not stuck_games.exists():
            self.stdout.write(
                self.style.SUCCESS('No stuck games found')
            )
            return
        
        self.stdout.write(
            f'Found {stuck_games.count()} stuck game(s) older than {timeout/60:.1f} minutes'
        )
        
        try:
            redis_client = redis.Redis.from_url(settings.REDIS_URL)
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to connect to Redis: {e}')
            )
            return
        
        cleaned_count = 0
        
        for game in stuck_games:
            time_stuck = timezone.now() - game.created_at
            self.stdout.write(
                f'Processing game {game.id} (stuck for {time_stuck.total_seconds()/60:.1f} minutes)'
            )
            
            if not dry_run:
                try:
                    # Clean up Redis locks and task state
                    lock_keys = [
                        f"game_creation_lock:{game.id}",
                        f"game_creation_task:{game.id}",
                        f"game_creation_status:{game.id}",
                    ]
                    
                    for key in lock_keys:
                        redis_client.delete(key)
                    
                    # Remove any Celery task result keys
                    task_keys = redis_client.keys(f"celery-task-meta-*{game.id}*")
                    if task_keys:
                        redis_client.delete(*task_keys)
                    
                    # Update game status to failed
                    game.status = 'failed'
                    game.error_message = f"Game creation timed out after {time_stuck.total_seconds()/60:.1f} minutes"
                    game.save(update_fields=['status', 'error_message'])
                    
                    cleaned_count += 1
                    self.stdout.write(
                        self.style.SUCCESS(f'  ✓ Cleaned up game {game.id}')
                    )
                    
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'  ✗ Failed to clean up game {game.id}: {e}')
                    )
            else:
                self.stdout.write(
                    f'  Would clean up game {game.id}'
                )
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING(f'DRY RUN: Would have cleaned up {stuck_games.count()} game(s)')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f'Successfully cleaned up {cleaned_count} game(s)')
            )