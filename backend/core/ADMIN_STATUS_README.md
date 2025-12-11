# Admin Status Monitoring Page

A comprehensive admin monitoring dashboard for the Eternalore platform that provides real-time health monitoring, error tracking, and service testing capabilities.

## Features

### 1. **System Health Monitoring**
- **CPU, Memory, and Disk Usage**: Real-time system resource monitoring with visual indicators
- **Database Health**: Connection status and response time monitoring
- **Redis Health**: Connection status, response time, and memory usage
- **Celery Workers**: Worker status and active task monitoring
- **External APIs**: OpenAI and Replicate API connectivity monitoring

### 2. **Error Tracking and Analysis**
- **Error Rate Visualization**: 24-hour error rate chart with hourly breakdown
- **Error Classification**: Automatic categorization of errors by type
- **Recent Error Log**: Quick access to recent error messages
- **Success/Failure Metrics**: Track turn processing success rates

### 3. **Game Activity Metrics**
- Active games count
- Games created in last hour/24 hours
- Turns processed in last hour/24 hours
- Active users in last 24 hours
- Average turn processing time

### 4. **Service Testing Suite**
Run comprehensive tests directly from the browser:
- Database connectivity test
- Redis connectivity test
- Celery worker test
- OpenAI API test
- Replicate API test
- User creation test
- Game creation test
- Turn processing test
- Media storage test

### 5. **Live Log Viewer**
- View main application logs
- View error logs
- Real-time log tailing with syntax highlighting

### 6. **Administrative Actions**
- Clear application cache
- Run individual or all tests
- Auto-refresh metrics every 30 seconds

## Access

1. **URL**: `/admin/status/`
2. **Authentication**: Requires staff privileges
3. **Test Admin Credentials** (for development):
   - Username: `admin`
   - Password: `adminpass123`

## API Endpoints

All endpoints require staff authentication:

- `GET /admin/api/health/` - Overall health status
- `GET /admin/api/system-metrics/` - System resource metrics
- `GET /admin/api/error-metrics/?hours=24` - Error metrics for specified hours
- `GET /admin/api/game-metrics/` - Game-related metrics
- `POST /admin/api/run-test/` - Run a specific test
- `POST /admin/api/run-all-tests/` - Run all tests
- `POST /admin/api/clear-cache/` - Clear application cache
- `GET /admin/api/log-tail/?type=main&lines=100` - Get recent log entries
- `GET /admin/api/service-details/?service=database` - Get detailed service info

## Implementation Details

### Architecture
- **Monitoring Module** (`user/monitoring.py`): Core monitoring logic and metrics collection
- **Admin Views** (`user/admin_views.py`): Staff-protected endpoints
- **Frontend** (`user/templates/admin_status.html`): Real-time dashboard with AJAX updates
- **Tests** (`user/test_monitoring.py`): Comprehensive test coverage

### Dependencies
- `psutil`: System resource monitoring
- `redis`: Redis connectivity
- `celery`: Worker monitoring
- Django's built-in staff authentication

### Security
- All endpoints protected by Django's `@staff_member_required` decorator
- CSRF protection on POST endpoints
- No sensitive data exposed in responses

## Usage

1. **Login as staff user**
2. **Navigate to** `/admin/status/`
3. **Monitor system health** - Check overall status badge
4. **Run tests** - Click "Run All Tests" to verify all services
5. **Check errors** - Review error chart and recent errors
6. **View logs** - Click on "Live Logs" tab

## Troubleshooting

### Common Issues

1. **404 on admin status page**
   - Ensure you're logged in as a staff user
   - Check that URLs are properly configured

2. **Tests failing**
   - Check environment variables (API keys)
   - Ensure all services (Redis, PostgreSQL) are running
   - Check log files for detailed error messages

3. **Metrics not updating**
   - Check browser console for JavaScript errors
   - Verify AJAX endpoints are accessible
   - Clear browser cache

### Log Locations
- Main log: `/backend/core/var/logs/Eternalore.log`
- Error log: `/backend/core/var/logs/error.log`

## Production Considerations

1. **Performance**: Consider caching metrics with longer TTL in production
2. **Security**: Restrict access to trusted IP addresses if needed
3. **Monitoring**: Set up alerts for critical errors
4. **Scaling**: Consider using dedicated monitoring services for large deployments

## Testing

Run the test suite:
```bash
python -m pytest user/test_monitoring.py -v
```

All 18 tests should pass, covering:
- Monitoring utilities
- Service testers
- Admin view endpoints
- Authentication requirements