from django.contrib import admin

from django.contrib import admin
from .models import User, GameCharacter, Game, GameStory, GameChapter, GameTurn

admin.site.register(User)
admin.site.register(GameCharacter)
admin.site.register(Game)
admin.site.register(GameStory)
admin.site.register(GameChapter)
admin.site.register(GameTurn)