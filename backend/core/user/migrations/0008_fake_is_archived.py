# Generated manually to fix migration state

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0007_game_is_archived_game_game_is_arch_940880_idx'),
    ]

    operations = [
        migrations.RunSQL(
            # This field already exists in the database from migration 0007
            # We're just updating Django's migration state
            sql=migrations.RunSQL.noop,
            reverse_sql=migrations.RunSQL.noop,
            state_operations=[
                migrations.AddField(
                    model_name='game',
                    name='is_archived',
                    field=models.BooleanField(db_index=True, default=False),
                ),
                migrations.AddIndex(
                    model_name='game',
                    index=models.Index(fields=['is_archived'], name='game_is_arch_940880_idx'),
                ),
            ],
        ),
    ]