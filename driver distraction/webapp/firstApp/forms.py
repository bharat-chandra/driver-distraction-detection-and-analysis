from django import forms
from .models import Performace
#DataFlair
class Form(forms.ModelForm):
    class Meta:
        model = Performace
        fields = '__all__'