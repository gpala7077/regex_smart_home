# Installation

Assumption: You have already installed **HACS**, **AppDaemon** and **MariaDB**.
1. If you do not have HACS installed, please follow the instructions here: https://hacs.xyz/docs/installation/manual.
2. If you do not have AppDaemon installed, please follow the instructions here:https://appdaemon.readthedocs.io/en/latest/INSTALL.html
3. If you do not have MariaDB installed, please follow the instructions here: https://github.com/home-assistant/addons/blob/master/mariadb/DOCS.md


This repository is not part of the default HACS store. To add it to your HACS, you need to add it as a
custom repository. To do this, go to the HACS settings and add the following URL as a custom repository and choose
'AppDaemon' as the category:

Enter this URL: https://github.com/gpala7077/regex_smart_home.git

It will look like this:
  <div style="display: flex; justify-content: space-around;">
  <div><img src="/apps/static/custom_repository.png" alt="Custom Repo" style="width: 50%; max-width: 500px;"/></div>
  </div>



# Configuration
This is best used when you have a standardized naming convention for your entities. 

It assumes the following naming convention:
domain.room_device_type
or 
domain_room_device_type_sensor

so for example:
- humidifier.living_room_humidifier
- sensor.living_room_humidifier_current_humidity

or
- binary_sensor.living_room_occupancy_1
- binary_sensor.living_room_occupancy_2

or 
- fan.living_room_purifier
- sensor.living_room_purifier_pm2_5


This app does not do anything it is a dependency for other apps.
