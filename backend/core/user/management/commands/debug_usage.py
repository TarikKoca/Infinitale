from django.core.management.base import BaseCommand
from django.utils import timezone
from decimal import Decimal
from user.models import User, UsageTracking

class Command(BaseCommand):
    help = 'Debug usage tracking for users'

    def add_arguments(self, parser):
        parser.add_argument('--email', type=str, help='User email to debug')

    def handle(self, *args, **options):
        email = options.get('email')
        
        if email:
            users = User.objects.filter(email=email)
        else:
            users = User.objects.all()
        
        for user in users:
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"User: {getattr(user, 'display_name', user.email.split('@')[0])} ({user.email})")
            self.stdout.write(f"Subscription: {user.subscription_plan}")
            self.stdout.write(f"Total Audio Generated (all-time): {user.total_audio_seconds_generated} seconds")
            
            # Get current month usage
            usage_stats = user.current_month_usage()
            self.stdout.write(f"\nCurrent Month Usage:")
            self.stdout.write(f"  - Audio seconds: {usage_stats.get('total_audio_seconds', 0)}")
            self.stdout.write(f"  - Text characters: {usage_stats.get('total_characters', 0)}")
            self.stdout.write(f"  - Turn count: {usage_stats.get('turn_count', 0)}")
            
            # Get all usage records for this month
            now = timezone.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            records = UsageTracking.objects.filter(
                user=user,
                created_at__gte=month_start
            ).order_by('-created_at')
            
            self.stdout.write(f"\nDetailed Usage Records (this month): {records.count()} records")
            for record in records[:5]:  # Show last 5 records
                self.stdout.write(f"  - {record.created_at}: {record.audio_duration_seconds}s, {record.text_characters} chars, type: {record.audio_type}")
                if record.game_turn:
                    self.stdout.write(f"    Game: {record.game_turn.game.name}, Turn: {record.game_turn.turn_number}")
            
            # Check if there are any UsageTracking records at all
            all_records = UsageTracking.objects.filter(user=user).count()
            self.stdout.write(f"\nTotal Usage Records (all-time): {all_records}")