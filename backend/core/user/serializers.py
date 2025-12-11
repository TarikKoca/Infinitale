from rest_framework import serializers
from user.models import Game, GameCharacter, GameStory, GameChapter, GameTurn

class GameSerializer(serializers.ModelSerializer):
    class Meta:
        model = Game
        fields = "__all__"

class GameCharacterSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameCharacter
        fields = "__all__"

class GameStorySerializer(serializers.ModelSerializer):
    class Meta:
        model = GameStory
        fields = "__all__"

class GameChapterSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameChapter
        fields = "__all__"

class GameTurnSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameTurn
        fields = "__all__"