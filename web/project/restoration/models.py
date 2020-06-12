from django.db import models

# Create your models here.
class Picture(models.Model):
    name = models.CharField(max_length=10)
    photo = models.ImageField(upload_to="%Y%m%d")
