# 🚀 GoonSteen - Plesk Deployment Guide

## ✅ Ready-to-Deploy Web Directory

This `/web` directory contains **everything you need** for your Plesk deployment. Simply drag and drop this entire folder to your Plesk server!

## 📁 Complete File Structure
```
web/
├── index.php                    # Main PHP entry point
├── config.php                   # Configuration settings
├── .htaccess                    # URL rewriting rules
├── composer.json                # PHP dependencies
├── database_schema.sql          # Database schema
├── test_backend.php            # Test script
├── PLESK_DEPLOYMENT_GUIDE.md   # This guide
├── classes/                    # PHP backend classes
│   ├── Database.php            # Database management
│   ├── Auth.php               # Authentication system
│   ├── Router.php             # URL routing
│   ├── UserController.php     # User API endpoints
│   ├── AdminController.php    # Admin API endpoints
│   └── DashboardController.php # Dashboard API endpoints
├── css/                       # Stylesheets
│   ├── styles.css
│   ├── auth.css
│   ├── dashboard.css
│   └── admin.css
├── js/                        # JavaScript files
│   ├── main.js
│   ├── auth.js
│   ├── dashboard.js
│   └── admin.js
├── logs/                      # Log directory (create if missing)
├── index.html                 # Landing page
├── login.html                 # Login page
├── signup.html                # Registration page
├── dashboard.html             # User dashboard
├── admin-dashboard.html       # Admin dashboard
├── admin-user-edit.html       # Admin user editor
├── test-auth.html            # Auth testing page
└── WEB_README.md             # Frontend documentation
```

## 🎯 Simple 3-Step Deployment

### Step 1: Upload Files
1. **Zip the entire `/web` directory**
2. **Upload to your Plesk domain's `httpdocs` folder**
3. **Extract the zip file**

### Step 2: Set Permissions
In Plesk File Manager:
- Set `classes/` folder to **755**
- Set `logs/` folder to **777** (create if missing)
- Set all `.php` files to **644**
- Set `.htaccess` to **644**

### Step 3: Configure PHP
In Plesk → PHP Settings:
- **PHP Version**: 7.4 or higher
- **Required Extensions**: PDO, PDO_SQLite, JSON, OpenSSL, cURL
- **Memory Limit**: 256M or higher

## 🧪 Test Your Deployment

### 1. Run the Test Script
Visit: `https://yourdomain.com/test_backend.php`

This will verify:
- ✅ Database connection
- ✅ Authentication system
- ✅ All PHP classes
- ✅ File permissions
- ✅ PHP configuration

### 2. Test API Endpoints
- **Health Check**: `https://yourdomain.com/api/health`
- **Main Site**: `https://yourdomain.com/`

### 3. Login as Admin
- **URL**: `https://yourdomain.com/login.html`
- **Username**: `admin`
- **Password**: `admin123`
- **⚠️ Change this password immediately!**

## 🔧 Configuration Options

### Environment Variables (Optional)
Set these in Plesk → PHP → Environment Variables:
```
JWT_SECRET=your-custom-jwt-secret-here
APP_ENV=production
```

### Database Location
The SQLite database will be created as:
`/web/web_database.db`

## 🛡️ Security Checklist

### After Deployment:
1. ✅ **Change admin password** from default `admin123`
2. ✅ **Set custom JWT_SECRET** in environment variables
3. ✅ **Enable HTTPS** in Plesk
4. ✅ **Set proper file permissions**
5. ✅ **Test all functionality**

## 📊 What's Included

### ✅ Complete Backend API
- **30+ API endpoints** identical to Python version
- **JWT authentication** system
- **User management** with registration/login
- **Admin panel** with full functionality
- **Dashboard** with real-time data
- **Betting system** with Kelly Criterion
- **Notification** system

### ✅ Frontend Interface
- **Responsive design** for all devices
- **Modern UI** with animations
- **Admin dashboard** for management
- **User dashboard** for betting
- **Authentication** pages

### ✅ Database System
- **SQLite database** (no external DB needed)
- **Auto-initialization** with schema
- **Pre-populated** with NBA teams
- **Default admin user** ready to use

## 🚨 Important Notes

### Database Security
- The database file will be created in the web root
- Consider moving it outside the web directory for production
- Regular backups are recommended

### Admin Access
- **Default username**: `admin`
- **Default password**: `admin123`
- **⚠️ CHANGE IMMEDIATELY** after first login

### File Permissions
```bash
# If using SSH/terminal access:
chmod -R 755 /path/to/web/classes/
chmod -R 777 /path/to/web/logs/
chmod 644 /path/to/web/*.php
chmod 644 /path/to/web/.htaccess
```

## 🔍 Troubleshooting

### Common Issues:

1. **500 Internal Server Error**
   - Check PHP error logs in Plesk
   - Verify file permissions
   - Ensure required PHP extensions are enabled

2. **Database Issues**
   - Ensure web directory is writable
   - Check if PDO SQLite extension is enabled
   - Look for errors in `logs/app.log`

3. **API Not Working**
   - Verify `.htaccess` is working
   - Test direct access: `/index.php?route=api/health`
   - Check URL rewriting is enabled

4. **Authentication Problems**
   - Verify JWT_SECRET is set
   - Check browser console for errors
   - Ensure cookies are enabled

### Debug Mode
To enable debug mode, set in Plesk PHP settings:
```
APP_ENV=development
```

## 📈 Performance Tips

### For Production:
1. **Enable OPcache** in PHP settings
2. **Use PHP 8.0+** for better performance
3. **Enable gzip compression**
4. **Set up SSL certificate**
5. **Configure caching headers**

## 🎉 Success Indicators

✅ **Test script passes all checks**
✅ **Main site loads without errors**
✅ **API health check returns JSON**
✅ **Admin login works**
✅ **User registration works**
✅ **Dashboard loads data**

## 📞 Support

If you encounter issues:
1. **Run the test script** first
2. **Check Plesk error logs**
3. **Verify PHP configuration**
4. **Test with debug mode enabled**

---

## 🎯 **YOU'RE ALL SET!**

Your NBA betting platform is now **100% ready for Plesk deployment**. The `/web` directory contains everything needed for a complete, functional website with zero additional setup required.

**Just drag, drop, and deploy!** 🏀
