from django.db import models

# Create your models here.
class Performance(models.Model):
    driver_id = models.IntegerField()
    date = models.TextField()
    timestamp = models.TextField()
    action = models.TextField()
    class Meta:
        db_table = 'Performance'