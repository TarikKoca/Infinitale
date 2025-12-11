from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("user", "0004_alter_gamestory_messages"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="subscription_plan",
            field=models.CharField(
                default="lite",
                choices=[("lite", "Lite"), ("plus", "Plus"), ("pro", "Pro")],
                max_length=8,
                db_index=True,
            ),
        ),
        migrations.AddIndex(
            model_name="user",
            index=models.Index(fields=["subscription_plan"], name="user_sub_plan_idx"),
        ),
    ]


