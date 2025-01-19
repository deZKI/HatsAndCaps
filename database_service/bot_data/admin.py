from django.contrib import admin

from .models import MessageHistory, User


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('telegram_id', 'username', 'created_at')
    search_fields = ('telegram_id', 'username')


@admin.register(MessageHistory)
class MessageHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'message', 'created_at', 'analysis_result')
    search_fields = ('user__telegram_id', 'message')
