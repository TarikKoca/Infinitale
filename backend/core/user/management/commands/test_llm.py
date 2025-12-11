from django.core.management.base import BaseCommand
import logging

class Command(BaseCommand):
    help = 'Test LLM functions for coroutine issues'

    def handle(self, *args, **options):
        logger = logging.getLogger(__name__)
        
        # Test importing and running llm_story multiple times
        self.stdout.write("Testing LLM functions...")
        
        try:
            # First import
            from user.prompts import llm_story
            self.stdout.write("First import successful")
            
            # Test call 1
            result1 = llm_story(
                genre="Fantasy",
                main_char="Test Hero",
                difficulty="medium",
                extra="",
                characters="Hero, Wizard, Knight"
            )
            self.stdout.write(f"First call successful: {bool(result1)}")
            
            # Test call 2 (this might fail if coroutine is reused)
            result2 = llm_story(
                genre="Sci-Fi", 
                main_char="Space Captain",
                difficulty="hard",
                extra="",
                characters="Captain, Robot, Alien"
            )
            self.stdout.write(f"Second call successful: {bool(result2)}")
            
            self.stdout.write(self.style.SUCCESS("✓ No coroutine reuse issues detected"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"✗ Error: {e}"))
            logger.exception("LLM test failed")