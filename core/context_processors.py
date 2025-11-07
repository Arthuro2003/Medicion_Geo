from .models import UserPreferences


def user_preferences(request):
  context = {
    'user_theme': 'light',
    'user_preferences': None
  }

  if request.user.is_authenticated:
    try:
      preferences = UserPreferences.get_or_create_for_user(request.user)
      context['user_theme'] = preferences.theme
      context['user_preferences'] = preferences
    except UserPreferences.DoesNotExist:
      pass

  return context
