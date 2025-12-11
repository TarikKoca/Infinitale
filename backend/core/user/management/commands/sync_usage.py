from django.core.management.base import BaseCommand
from django.utils import timezone
from decimal import Decimal
from user.models import User, UsageTracking

class Command(BaseCommand):
    help = 'Sync historical usage data for users without UsageTracking records'

    def handle(self, *args, **options):
        users = User.objects.all()
        
        for user in users:
            # Check if user has any usage tracking records
            existing_total = UsageTracking.objects.filter(user=user).count()
            
            if existing_total == 0 and user.total_audio_seconds_generated > 0:
                # User has historical usage but no tracking records
                # Create a single record for the current month to represent historical usage
                self.stdout.write(f"Creating historical usage record for {user.email}")
                self.stdout.write(f"  Total audio generated: {user.total_audio_seconds_generated} seconds")
                
                UsageTracking.objects.create(
                    user=user,
                    game_turn=None,  # No associated turn since games were deleted
                    audio_duration_seconds=user.total_audio_seconds_generated,
                    text_characters=0,  # Unknown
                    audio_type='narration',
                    created_at=timezone.now()
                )
                
                self.stdout.write(self.style.SUCCESS(f"  Created historical usage record"))
            elif existing_total > 0:
                # Check if tracked usage matches total
                from django.db.models import Sum
                tracked_total = UsageTracking.objects.filter(user=user).aggregate(
                    total=Sum('audio_duration_seconds')
                )['total'] or Decimal('0')
                
                diff = user.total_audio_seconds_generated - tracked_total
                
                if diff > 0.01:  # Small tolerance for rounding
                    self.stdout.write(f"Usage mismatch for {user.email}")
                    self.stdout.write(f"  Total field: {user.total_audio_seconds_generated}")
                    self.stdout.write(f"  Tracked sum: {tracked_total}")
                    self.stdout.write(f"  Difference: {diff}")
                    
                    # Create a reconciliation record
                    UsageTracking.objects.create(
                        user=user,
                        game_turn=None,
                        audio_duration_seconds=diff,
                        text_characters=0,
                        audio_type='narration',
                        created_at=timezone.now()
                    )
                    self.stdout.write(self.style.SUCCESS(f"  Created reconciliation record for {diff} seconds"))
                else:
                    self.stdout.write(f"Usage tracking is in sync for {user.email}")