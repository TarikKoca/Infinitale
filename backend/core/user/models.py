import uuid
import logging
from decimal import Decimal
from django.db import models
from django.core.validators import MinValueValidator
from django.utils import timezone

logger = logging.getLogger(__name__)


# --- Game Constants (Merged from JSON files) ---
class GameConstants:
    """All game constants merged directly into code - no JSON files needed."""
    
    MALE_NAMES = [
        "Liam", "Noah", "Oliver", "Elijah", "James", "William", "Benjamin", "Lucas", "Henry", "Alexander",
        "Mason", "Michael", "Ethan", "Daniel", "Jacob", "Logan", "Jackson", "Levi", "Sebastian", "Mateo",
        "Jack", "Owen", "Theodore", "Aiden", "Samuel", "Joseph", "John", "David", "Wyatt", "Matthew",
        "Luke", "Asher", "Carter", "Julian", "Grayson", "Leo", "Jayden", "Gabriel", "Isaac", "Lincoln",
        "Anthony", "Hudson", "Dylan", "Ezra", "Thomas", "Charles", "Christopher", "Jaxon", "Maverick", "Josiah",
        "Isaiah", "Andrew", "Elias", "Joshua", "Nathan", "Caleb", "Ryan", "Adrian", "Miles", "Eli",
        "Nolan", "Christian", "Aaron", "Cameron", "Ezekiel", "Colton", "Luca", "Landon", "Hunter", "Jonathan",
        "Santiago", "Axel", "Easton", "Cooper", "Jeremiah", "Angel", "Roman", "Connor", "Jameson", "Robert",
        "Greyson", "Jordan", "Ian", "Carson", "Jaxson", "Leonardo", "Nicholas", "Dominic", "Austin", "Everett",
        "Brooks", "Xavier", "Kai", "Jose", "Parker", "Adam", "Jace", "Wesley", "Kayden", "Silas",
        "Bentley", "Declan", "Weston", "Micah", "Ayden", "Diego", "Vincent", "Ryder", "Max", "Malachi",
        "Emmett", "Sawyer", "Beau", "Harrison", "Kaleb", "Bryson", "Maxwell", "Zayden", "Justin", "Braxton",
        "Ryker", "George", "Kaiden", "Ivan", "Carlos", "Kingston", "Maddox", "Juan", "Ashton", "Jayce",
        "Kevin", "Judah", "Giovanni", "Eric", "Brayden", "Alan", "Patrick", "Joel", "Victor", "Abel",
        "Alex", "Arthur", "Abraham", "Nicolas", "Jesse", "Blake", "Finn", "Brantley", "Brody", "Jake",
        "Emiliano", "Zachary", "Knox", "Matteo", "Miguel", "Hayden", "Gael", "Rowan", "Israel", "Nathaniel",
        "Messiah", "Theo", "Beckett", "Adriel", "Enzo", "Dean", "Emmanuel", "Antonio", "Karter", "Ellis",
        "Grant", "Caden", "Stephen", "Kyrie", "Francisco", "Shane", "Erick", "Shawn", "Walter", "Martin",
        "Oscar", "Archer", "Andres", "Dallas", "Tobias", "Jaden", "Colt", "Jorge", "Zion", "Andy",
        "Johnny", "Marco", "Spencer", "Travis", "Eduardo", "Chance", "Ricardo", "Walter", "Pedro", "Julius",
        "Dakota", "Manuel", "Rafael", "Maximus", "Ali", "Reid", "Ronan", "Kaden", "Tucker", "Bodhi",
        "Gideon", "Nash", "Major", "Walker", "Dawson", "Arlo", "Clayton", "Malik", "Prince", "Kash",
        "Donovan", "Seth", "Allen", "Tate", "Frederick", "Ari", "Desmond", "Brady", "Raiden", "Wade",
        "Otto", "Porter", "Jonah", "Iker", "Remington", "Cody", "Warren", "Quinn", "Hector", "Kamden",
        "Anderson", "Paxton", "Bruno", "Colin", "Jalen", "Finley", "Bryce", "Kellen", "Zander", "Lawson",
        "Derek", "Rory", "Ronin", "Keegan", "Samson", "Kyson", "Cyrus", "Corbin", "Jett", "Troy",
        "Gunnar", "Kobe", "Cassius", "Mohamed", "Bo", "Skyler", "Emanuel", "Reed", "Matias", "Ezequiel"
    ]
    
    FEMALE_NAMES = [
        "Olivia", "Emma", "Charlotte", "Amelia", "Ava", "Sophia", "Isabella", "Mia", "Evelyn", "Harper",
        "Luna", "Camila", "Gianna", "Elizabeth", "Eleanor", "Ella", "Abigail", "Sofia", "Avery", "Scarlett",
        "Emily", "Aria", "Penelope", "Chloe", "Layla", "Mila", "Nora", "Hazel", "Madison", "Ellie",
        "Lily", "Nova", "Isla", "Grace", "Violet", "Aurora", "Riley", "Zoey", "Willow", "Emilia",
        "Stella", "Zoe", "Victoria", "Hannah", "Addison", "Leah", "Lucy", "Eliana", "Ivy", "Everly",
        "Lillian", "Paisley", "Elena", "Naomi", "Maya", "Natalie", "Kinsley", "Delilah", "Claire", "Audrey",
        "Aaliyah", "Ruby", "Brooklyn", "Alice", "Aubrey", "Autumn", "Leilani", "Savannah", "Valentina", "Kennedy",
        "Madelyn", "Josephine", "Bella", "Skylar", "Genesis", "Sophie", "Hailey", "Sadie", "Natalia", "Quinn",
        "Caroline", "Allison", "Gabriella", "Anna", "Serenity", "Nevaeh", "Cora", "Ariana", "Emery", "Lydia",
        "Jade", "Sarah", "Eva", "Adeline", "Madeline", "Piper", "Rylee", "Athena", "Peyton", "Everleigh",
        "Maria", "Clara", "Rose", "Daisy", "Lucia", "Julia", "Ashley", "Vivian", "Hadley", "Gianna",
        "Eloise", "Raelynn", "Sage", "Lyla", "Juniper", "Alina", "Adalynn", "Amara", "Adriana", "Arianna",
        "Faith", "Molly", "Ana", "Kaylee", "Annabelle", "Melody", "Lyla", "Margaret", "Isabel", "Jasmine",
        "Brielle", "Parker", "Lyla", "Eliza", "Arya", "Diana", "Reese", "Daniela", "Alexandra", "Katherine",
        "Andrea", "Gabriela", "Cecilia", "Juliana", "Rebecca", "Iris", "Ayla", "Rosalie", "Eden", "Mckenzie",
        "Laila", "Andrea", "Remi", "Melody", "Amara", "Rosalie", "Sydney", "Jessica", "Aliyah", "Izabella",
        "Callie", "Haven", "Anya", "Aria", "Miriam", "Anastasia", "Alaina", "Lyric", "Dakota", "Brianna",
        "Briella", "Teagan", "Dahlia", "Lilith", "Heidi", "Daniela", "Summer", "Marley", "Rachel", "Brooke",
        "Rowan", "Michelle", "Paula", "Harmony", "Sutton", "Leia", "Mckenna", "Adalyn", "Zuri", "Hope",
        "Lola", "Alana", "Gemma", "Kenzie", "Harlow", "Journey", "Talia", "Juliette", "Ember", "Carmen",
        "Miranda", "Elise", "Dakota", "Destiny", "Ophelia", "Delaney", "Mya", "River", "Arielle", "Millie",
        "Phoenix", "Sienna", "Vera", "Camille", "Paige", "Adriana", "Alivia", "London", "Kate", "Rosemary",
        "Kathryn", "Mabel", "Kylie", "Cassidy", "Raegan", "Scarlet", "Amy", "Nylah", "Trinity", "Skyler",
        "Esther", "Blake", "Hallie", "Ruth", "Saylor", "Lucille", "Kaitlyn", "Vivienne", "Joanna", "Magnolia",
        "Kendall", "Jordyn", "Melody", "Sawyer", "Daphne", "Phoebe", "Isla", "Noelle", "Selena", "Dakota"
    ]
    
    MALE_VOICES = ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"]
    FEMALE_VOICES = ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"]
    
    SOUND_EFFECTS = [
        {"name": "footstep_stone.wav", "description": "Walking on stone floor."},
        {"name": "footstep_wood.wav", "description": "Walking on wooden floor."},
        {"name": "footstep_grass.wav", "description": "Walking on grass."},
        {"name": "footstep_metal.wav", "description": "Walking on metal surface."},
        {"name": "door_creak.wav", "description": "Old wooden door opening slowly."},
        {"name": "door_slam.wav", "description": "Door shutting with force."},
        {"name": "knock_wood.wav", "description": "Knocking on wooden door."},
        {"name": "crowd_murmur.wav", "description": "Background crowd talking."},
        {"name": "sword_clash.wav", "description": "Metal weapons clashing."},
        {"name": "gun_shot.wav", "description": "Single gunfire shot."},
        {"name": "gun_reload.wav", "description": "Reloading a firearm."},
        {"name": "explosion_small.wav", "description": "Small explosion or grenade."},
        {"name": "explosion_large.wav", "description": "Massive explosion blast."},
        {"name": "glass_break.wav", "description": "Shattering glass."},
        {"name": "rain_light.wav", "description": "Gentle rainfall."},
        {"name": "rain_heavy.wav", "description": "Heavy rainfall with thunder."},
        {"name": "thunder_strike.wav", "description": "Loud thunder crash."},
        {"name": "wind_howl.wav", "description": "Soft wind blowing."},
        {"name": "river_stream.wav", "description": "Flowing river water."},
        {"name": "ocean_wave.wav", "description": "Ocean waves hitting shore."},
        {"name": "fire_crackle.wav", "description": "Campfire or fireplace burning."},
        {"name": "forest_birds.wav", "description": "Birds chirping in forest."},
        {"name": "city_traffic.wav", "description": "Busy street with cars."},
        {"name": "dog_bark.wav", "description": "Dog barking."},
        {"name": "cat_meow.wav", "description": "Cat meowing."},
        {"name": "wolf_howl.wav", "description": "Wolf howling."},
        {"name": "monster_roar.wav", "description": "Creature roar."},
    ]
    @classmethod
    def get_names(cls):
        """Get all names by gender."""
        return {"male": cls.MALE_NAMES, "female": cls.FEMALE_NAMES}
    
    @classmethod
    def get_voices(cls):
        """Get all voices by gender."""
        return {"male": cls.MALE_VOICES, "female": cls.FEMALE_VOICES}
    
    @classmethod
    def get_sound_effects(cls):
        """Get all sound effects."""
        return cls.SOUND_EFFECTS
    
    @classmethod
    def get_male_names(cls):
        """Get male names."""
        return cls.MALE_NAMES
    
    @classmethod
    def get_female_names(cls):
        """Get female names."""
        return cls.FEMALE_NAMES
    
    @classmethod
    def get_male_voices(cls):
        """Get male voices."""
        return cls.MALE_VOICES
    
    @classmethod
    def get_female_voices(cls):
        """Get female voices."""
        return cls.FEMALE_VOICES
    
    # Visual Style Enhancement Mappings (suffix-only to preserve original description priority)
    VISUAL_STYLE_ENHANCEMENTS = {
        "realistic": ", photorealistic, ultra-detailed, 8k resolution, professional photography, cinematic lighting, sharp focus, depth of field, masterpiece quality, physically accurate, high dynamic range, professional color grading, natural lighting",
        "illustration": ", digital art, concept art, detailed illustration, artstation quality, painted style, artistic composition, vibrant colors, professional illustration, clean brushwork, artistic lighting, digital painting masterpiece",
        "anime": ", anime style, manga art, cel shading, Japanese animation style, clean lineart, detailed anime art, vibrant anime colors, smooth cel shading, professional anime quality, studio animation level",
        "pixel_art": ", pixel art, 16-bit style, retro gaming art, detailed pixel graphics, crisp pixels, pixel perfect, retro aesthetic, classic game art style, clean pixel work, nostalgic gaming feel"
    }
    
    @classmethod
    def get_visual_style_enhancement(cls, visual_style: str) -> str:
        """Get visual style enhancement suffix for given style."""
        return cls.VISUAL_STYLE_ENHANCEMENTS.get(visual_style, "")


# --- Choice Enums ---

class VisualStyleChoices(models.TextChoices):
    REALISTIC = "realistic", "Realistic"
    PIXEL_ART = "pixel_art", "Pixel Art"
    ILLUSTRATION = "illustration", "Illustration"
    ANIME = "anime", "Anime"

class DifficultyChoices(models.TextChoices):
    EASY = "easy", "Easy"
    NORMAL = "normal", "Normal"
    HARD = "hard", "Hard"

class GenderChoices(models.TextChoices):
    MALE = "male", "Male"
    FEMALE = "female", "Female"
    UNSPECIFIED = "unspecified", "Unspecified"

class GameStatusChoices(models.TextChoices):
    CREATING = "creating", "Creating"
    ONGOING = "ongoing", "Ongoing"
    COMPLETED = "completed", "Completed"
    ERROR = "error", "Error"
    FAILED = "failed", "Failed"

class GameErrorType(models.TextChoices):
    CREATION = "creation", "Game Creation"
    TURN_PROCESSING = "turn_processing", "Turn Processing"
    MEDIA_GENERATION = "media_generation", "Media Generation"
    API_LIMIT = "api_limit", "API Limit"
    UNKNOWN = "unknown", "Unknown Error"

class StoryLengthChoices(models.TextChoices):
    SHORT = "short", "Short Story"
    STANDARD = "standard", "Full Adventure"

class TTSVoiceChoices(models.TextChoices):
    # Male voices
    AM_ADAM = "am_adam", "Adam"
    AM_ECHO = "am_echo", "Echo" 
    AM_ERIC = "am_eric", "Eric"
    AM_FENRIR = "am_fenrir", "Fenrir"
    AM_LIAM = "am_liam", "Liam"
    AM_MICHAEL = "am_michael", "Michael"
    AM_ONYX = "am_onyx", "Onyx"
    AM_PUCK = "am_puck", "Puck"
    AM_SANTA = "am_santa", "Santa"
    # Female voices
    AF_HEART = "af_heart", "Heart"
    AF_ALLOY = "af_alloy", "Alloy"
    AF_AOEDE = "af_aoede", "Aoede"
    AF_BELLA = "af_bella", "Bella"
    AF_JESSICA = "af_jessica", "Jessica"
    AF_KORE = "af_kore", "Kore"
    AF_NICOLE = "af_nicole", "Nicole"
    AF_NOVA = "af_nova", "Nova"
    AF_RIVER = "af_river", "River"
    AF_SARAH = "af_sarah", "Sarah"
    AF_SKY = "af_sky", "Sky"


# --- Base Models ---

class User(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=15)
    email = models.EmailField(unique=True)
    
    # Theme preference
    theme = models.CharField(
        max_length=10,
        choices=[('system', 'System'), ('light', 'Light'), ('dark', 'Dark')],
        default='system',
        help_text="User's preferred UI theme"
    )
    
    # Usage tracking fields
    total_audio_seconds_generated = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        default=Decimal("0.00"),
        help_text="Total seconds of audio generated all-time"
    )
    # Preferred UI/content language
    preferred_language = models.CharField(
        max_length=5,
        choices=[('en','English'),('es','Español'),('fr','Français'),('pt-br','Português (Brasil)')],
        default='en',
        db_index=True,
        help_text="User's preferred language for UI and content"
    )
    
    def current_month_usage(self):
        """Get current month's usage statistics."""
        from django.db.models import Sum, Count
        now = timezone.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        result = UsageTracking.objects.filter(
            user=self,
            created_at__gte=month_start
        ).aggregate(
            total_audio_seconds=Sum('audio_duration_seconds'),
            total_characters=Sum('text_characters'),
            turn_count=Count('id')
        )
        # Ensure we return 0 instead of None
        return {
            'total_audio_seconds': result.get('total_audio_seconds') or 0,
            'total_characters': result.get('total_characters') or 0,
            'turn_count': result.get('turn_count') or 0
        }
    
    class SubscriptionPlan(models.TextChoices):
        FREE = "free", "Free"
        LITE = "lite", "Lite"
        PLUS = "plus", "Plus"
        PRO = "pro", "Pro"
    subscription_plan = models.CharField(
        max_length=8,
        choices=SubscriptionPlan.choices,
        default=SubscriptionPlan.LITE,
        db_index=True,
    )
    # Billing state: whether the recurring subscription was canceled (access may remain until end of cycle)
    billing_is_canceled = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Set to True when provider notifies subscription canceled/past_due; reset on activation/renewal"
    )
    # For paid plans, track when the current billing cycle started.
    # Free plan ignores this and uses calendar month.
    plan_cycle_started_at = models.DateTimeField(null=True, blank=True, db_index=True)
    # Track last known paid plan and when its cycle started, to drive effective limits when toggling plans.
    last_paid_plan = models.CharField(
        max_length=8,
        choices=SubscriptionPlan.choices,
        null=True,
        blank=True,
        db_index=True,
        help_text="Most recent paid plan (lite/plus/pro) for limit calculations when on FREE"
    )
    last_paid_cycle_started_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Start time of the most recent paid billing cycle"
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "user"
        indexes = [
            models.Index(fields=["created_at"]),
            models.Index(fields=["subscription_plan"], name="user_sub_plan_idx"),
        ]

    def __str__(self):
        return f"{self.name} <{self.email}>"

    @property
    def display_name(self) -> str:
        """Display local-part of the email (before @). Falls back to name/email."""
        try:
            if self.email:
                local = (self.email or "").split("@")[0].strip()
                if local:
                    return local
        except Exception:
            pass
        return (self.name or self.email or "Player").split("@")[0]

    def save(self, *args, **kwargs):
        is_new = self.pk is None
        old_plan = None
        if not is_new:
            old_plan = User.objects.filter(pk=self.pk).values_list('subscription_plan', flat=True).first()
        
        super().save(*args, **kwargs)
        
        if is_new:
            logger.info(f"New user created: {self.email} with plan {self.subscription_plan}")
        elif old_plan and old_plan != self.subscription_plan:
            logger.info(f"User {self.email} subscription changed from {old_plan} to {self.subscription_plan}")

    # ---------- Usage window helpers ----------
    @property
    def has_paid_plan(self) -> bool:
        return self.subscription_plan in {
            self.SubscriptionPlan.LITE,
            self.SubscriptionPlan.PLUS,
            self.SubscriptionPlan.PRO,
        }

    @property
    def can_delete_account(self) -> bool:
        """Account can be deleted if user is on FREE or the paid subscription has been canceled.
        We intentionally allow deletion during the grace period after cancelation.
        """
        return (not self.has_paid_plan) or bool(self.billing_is_canceled)
    def get_usage_window_start(self):
        """Return the start datetime for usage tracking window.
        - Free plan: beginning of current calendar month
        - Paid plans: plan_cycle_started_at if set, else beginning of current month
        """
        now = timezone.now()
        if self.subscription_plan == self.SubscriptionPlan.FREE:
            # If user had a paid plan that started this calendar month, align window to that start
            try:
                month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if self.last_paid_cycle_started_at and self.last_paid_cycle_started_at >= month_start:
                    return self.last_paid_cycle_started_at
            except Exception:
                pass
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Paid plan
        if self.plan_cycle_started_at:
            return self.plan_cycle_started_at
        # Sensible fallback for existing users without a cycle start
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def get_effective_plan_for_limits(self) -> str:
        """Return the plan to use for game/deletion limits to avoid toggle exploits.
        Rules:
        - If currently on a paid plan, use it.
        - If currently FREE, but had a paid plan that started this calendar month, use that last paid plan.
        - Otherwise, use FREE.
        """
        try:
            if self.subscription_plan in {
                self.SubscriptionPlan.LITE,
                self.SubscriptionPlan.PLUS,
                self.SubscriptionPlan.PRO,
            }:
                return self.subscription_plan
            # Currently FREE: check last paid within calendar month
            now = timezone.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if (
                (self.last_paid_plan in {self.SubscriptionPlan.LITE, self.SubscriptionPlan.PLUS, self.SubscriptionPlan.PRO})
                and self.last_paid_cycle_started_at
                and self.last_paid_cycle_started_at >= month_start
            ):
                return self.last_paid_plan
        except Exception:
            pass
        return self.SubscriptionPlan.FREE

    def current_cycle_usage(self):
        """Get usage statistics within the current cycle window.
        Mirrors current_month_usage but respects plan cycle for paid users.
        """
        from django.db.models import Sum, Count
        window_start = self.get_usage_window_start()
        result = UsageTracking.objects.filter(
            user=self,
            created_at__gte=window_start
        ).aggregate(
            total_audio_seconds=Sum('audio_duration_seconds'),
            total_characters=Sum('text_characters'),
            turn_count=Count('id')
        )
        return {
            'total_audio_seconds': result.get('total_audio_seconds') or 0,
            'total_characters': result.get('total_characters') or 0,
            'turn_count': result.get('turn_count') or 0
        }


class GameCharacter(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    name = models.CharField(max_length=20)
    gender = models.CharField(
        max_length=12,
        choices=GenderChoices.choices,
        default=GenderChoices.UNSPECIFIED,
    )
    # Base64 yerine URL: depoda tutup burada yol/URL saklayın
    reference_image_url = models.URLField(blank=True, null=True)
    # Zorunlu base64 isterseniz: reference_image_b64 = models.TextField(blank=True, null=True)

    tts_voice = models.CharField(
        max_length=15,
        choices=TTSVoiceChoices.choices,
        default=TTSVoiceChoices.AF_ALLOY,
    )

    class Meta:
        db_table = "game_character"

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        is_new = self.pk is None
        super().save(*args, **kwargs)
        if is_new:
            logger.debug(f"New game character created: {self.name} with voice {self.tts_voice}")


class Game(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    user = models.ForeignKey(
        User,
        related_name="games",
        on_delete=models.CASCADE,
        db_index=True,
    )
    
    genre_name = models.CharField(max_length=15)  # Örn: mystery/thriller
    main_character_name = models.CharField(max_length=20)

    visual_style = models.CharField(
        max_length=15,
        choices=VisualStyleChoices.choices,
        default=VisualStyleChoices.ILLUSTRATION,
    )
    difficulty = models.CharField(
        max_length=10,
        choices=DifficultyChoices.choices,
        default=DifficultyChoices.NORMAL,
    )
    story_length = models.CharField(
        max_length=10,
        choices=StoryLengthChoices.choices,
        default=StoryLengthChoices.STANDARD,
    )
    # Lock game narrative/audio language at creation for consistency
    language_code = models.CharField(
        max_length=5,
        choices=[('en','English'),('es','Español'),('fr','Français'),('pt-br','Português (Brasil)')],
        default='en',
        db_index=True,
    )

    # Persist user's additional requests/preferences from new_game form
    extra_requests = models.TextField(blank=True, default="", db_default="")

    characters = models.ManyToManyField(
        GameCharacter,
        related_name="games",
        blank=True,
    )

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    last_played_at = models.DateTimeField(auto_now=True, db_index=True)

    status = models.CharField(
        max_length=10,
        choices=GameStatusChoices.choices,
        default=GameStatusChoices.CREATING,
        db_index=True,
    )
    
    is_archived = models.BooleanField(default=False, db_index=True)
    
    # Error tracking fields
    error_count = models.PositiveIntegerField(
        default=0,
        db_default=0,
        help_text="Total number of errors encountered"
    )
    last_error_at = models.DateTimeField(null=True, blank=True, help_text="Timestamp of last error")
    last_error_type = models.CharField(
        max_length=20,
        choices=GameErrorType.choices,
        null=True,
        blank=True,
        help_text="Type of the last error"
    )
    last_error_message = models.TextField(
        blank=True,
        default="",
        db_default="",
        help_text="Last error message for debugging"
    )
    is_error_critical = models.BooleanField(
        default=False,
        db_default=False,
        db_index=True,
        help_text="Whether the game has critical errors that prevent normal operation"
    )

    class Meta:
        db_table = "game"
        indexes = [
            models.Index(fields=["status", "last_played_at"]),
            models.Index(fields=["is_archived"]),
            models.Index(fields=["is_error_critical"], name="game_error_critical_idx"),
            models.Index(fields=["last_error_at"], name="game_last_error_time_idx"),
        ]

    def __str__(self):
        return f"Game {self.id} ({self.user_id})"

    def delete(self, *args, **kwargs):
        """Delete game and all associated media files."""
        from user.security import secure_delete_media_url
        
        logger.info(f"Deleting game {self.id} and its media files")
        
        # Collect all media URLs before deletion
        media_urls = []
        
        # Get character reference images
        for character in self.characters.all():
            if character.reference_image_url:
                media_urls.append(character.reference_image_url)
        
        # Get all turn media (TTS audio and generated images)
        for chapter in self.chapters.all():
            for turn in chapter.turns.all():
                if turn.generated_tts_url:
                    media_urls.append(turn.generated_tts_url)
                if turn.generated_visual_url:
                    media_urls.append(turn.generated_visual_url)
        
        # Delete the game (cascades to related objects)
        super().delete(*args, **kwargs)
        
        # Delete all collected media files
        deleted_count = 0
        for url in media_urls:
            try:
                if secure_delete_media_url(url):
                    deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete media file {url}: {e}")
        
        logger.info(f"Game {self.id} deleted. Removed {deleted_count}/{len(media_urls)} media files")
    
    def save(self, *args, **kwargs):
        is_new = self.pk is None
        old_status = None
        old_archived = None
        
        # Defensive: ensure TextFields never persist as NULL
        if getattr(self, "extra_requests", None) is None:
            self.extra_requests = ""
        
        if not is_new:
            old_values = Game.objects.filter(pk=self.pk).values('status', 'is_archived').first()
            if old_values:
                old_status = old_values['status']
                old_archived = old_values['is_archived']
        
        super().save(*args, **kwargs)
        
        if is_new:
            logger.info(f"New game created: {self.id} for user {self.user_id} - genre: {self.genre_name}, character: {self.main_character_name}")
        else:
            if old_status and old_status != self.status:
                logger.info(f"Game {self.id} status changed from {old_status} to {self.status}")
            if old_archived is not None and old_archived != self.is_archived:
                action = "archived" if self.is_archived else "unarchived"
                logger.info(f"Game {self.id} was {action}")
    
    def record_error(self, error_type, error_message, is_critical=False):
        """Record an error for this game."""
        self.error_count += 1
        self.last_error_at = timezone.now()
        self.last_error_type = error_type
        self.last_error_message = error_message[:1000]  # Limit message length
        if is_critical:
            self.is_error_critical = True
            self.status = GameStatusChoices.ERROR
        self.save(update_fields=[
            'error_count', 'last_error_at', 'last_error_type',
            'last_error_message', 'is_error_critical', 'status'
        ])
        logger.error(f"Game {self.id} error recorded: {error_type} - {error_message[:200]}")
    
    def clear_error_status(self):
        """Clear critical error status when game is recovered."""
        if self.is_error_critical:
            self.is_error_critical = False
            if self.status == GameStatusChoices.ERROR:
                self.status = GameStatusChoices.ONGOING
            self.save(update_fields=['is_error_critical', 'status'])
            logger.info(f"Game {self.id} error status cleared")


class GameStory(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    game = models.OneToOneField(
        Game,
        related_name="story",
        on_delete=models.CASCADE,
    )

    introductory_text = models.TextField()   # Dünya açıklaması / uzun giriş
    story = models.TextField()             # LLM plan
    realized_story = models.TextField()            # Oynanmış/gerçekleşen öykü
    # Mesaj geçmişi liste olarak tutulur: [{role, content}, ...]
    messages = models.JSONField(default=list, blank=True)
    ongoing_chapter_index = models.IntegerField(
        default=0, validators=[MinValueValidator(0)]
    )

    class Meta:
        db_table = "game_story"


class GameChapter(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    game = models.ForeignKey("user.Game", on_delete=models.CASCADE, related_name="chapters")
    index = models.PositiveIntegerField()  # 0,1,2...
    planned_chapter_story = models.TextField()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["game", "index"], name="uniq_game_chapter_index"),
        ]

    def __str__(self):
        return f"Chapter {self.index} of Game {self.game_id}"

    def save(self, *args, **kwargs):
        is_new = self.pk is None
        super().save(*args, **kwargs)
        if is_new:
            logger.debug(f"New chapter created: index {self.index} for game {self.game_id}")


class TurnStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    PROCESSING = "processing", "Processing"
    DONE = "done", "Done"
    FAILED = "failed", "Failed"

class GameTurn(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chapter = models.ForeignKey("user.GameChapter", on_delete=models.CASCADE, related_name="turns")
    index = models.PositiveIntegerField()
    status = models.CharField(max_length=12, choices=TurnStatus.choices, default=TurnStatus.PENDING, db_index=True)

    # izleme
    reserved_at = models.DateTimeField(default=timezone.now, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    attempts = models.PositiveSmallIntegerField(default=0)

    # istek / idempotensi
    user_text = models.TextField(blank=True, default="")
    idempotency_key = models.CharField(max_length=64, null=True, blank=True, unique=True)

    # üretim sonuçları
    new_lines = models.TextField(blank=True, default="")
    choices = models.JSONField(blank=True, default=list)
    generated_tts_url = models.URLField(blank=True, default="")
    generated_visual_url = models.URLField(blank=True, default="")
    selected_sound_effect = models.TextField(blank=True, default="")
    
    # Audio duration tracking (for quota management)
    audio_duration_seconds = models.DecimalField(
        max_digits=8, 
        decimal_places=2, 
        default=Decimal("0.00"),
        help_text="Duration of generated audio in seconds"
    )

    # hata
    error_message = models.TextField(blank=True, default="")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["chapter", "index"], name="uniq_chapter_index"),
        ]
        indexes = [
            models.Index(fields=["chapter", "status"]),
        ]

    def __str__(self):
        return f"Turn {self.index} (Chapter {self.chapter_id})"

    def save(self, *args, **kwargs):
        is_new = self.pk is None
        old_status = None
        
        if not is_new:
            old_status = GameTurn.objects.filter(pk=self.pk).values_list('status', flat=True).first()
        
        super().save(*args, **kwargs)
        
        if is_new:
            logger.debug(f"New turn created: {self.id} - chapter {self.chapter_id}, index {self.index}")
        elif old_status and old_status != self.status:
            logger.info(f"Turn {self.id} status changed from {old_status} to {self.status}")
            if self.status == TurnStatus.FAILED:
                logger.error(f"Turn {self.id} failed with error: {self.error_message[:200]}")


class UsageTracking(models.Model):
    """Track audio generation usage per turn for quota management."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="usage_records")
    game_turn = models.OneToOneField(GameTurn, on_delete=models.CASCADE, related_name="usage_tracking", null=True, blank=True)
    
    # Audio tracking
    audio_duration_seconds = models.DecimalField(
        max_digits=8, 
        decimal_places=2, 
        default=Decimal("0.00"),
        help_text="Duration of generated audio in seconds"
    )
    text_characters = models.PositiveIntegerField(
        default=0,
        help_text="Number of characters in the text that was converted to audio"
    )
    audio_type = models.CharField(
        max_length=20,
        choices=[
            ('narration', 'Narration'),
            ('character', 'Character'),
        ],
        default='narration'
    )
    
    # Tracking metadata
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    
    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["created_at"]),
        ]
    
    def __str__(self):
        return f"Usage {self.user.email}: {self.audio_duration_seconds}s on {self.created_at}"


class GameDeletionLog(models.Model):
    """Track per-user game deletions to enforce deletion caps per billing window."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="deletion_logs")
    # Store deleted game id as UUID (not FK) to preserve log after deletion
    deleted_game_id = models.UUIDField()
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "game_deletion_log"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"], name="del_user_created_idx"),
        ]

    def __str__(self):
        return f"Deletion by {self.user.email} at {self.created_at} (game {self.deleted_game_id})"