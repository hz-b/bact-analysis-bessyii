"""

Which interface is solving which problem!

* :class:`space_mapping.SpaceMappingInterface` map values from e.g. engineering to pyhsics and vice versa
  think of both as a space/ set (DEutsch Menge) where one maps from set `a` to set `b` and so one

* :class:`device_location.DeviceLocationServiceInterface`: from device names to position names

* :class:`standard_naming.StandardLocationNameInterface`: get from an abstact naming (similar to MML) to the one
  used at this installation and vice versa
"""


__all__ = ["name_index_mapping", "standard_naming", "space_mapping", "geographic_information_system", "device_location"]