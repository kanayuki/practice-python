# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-20.55.25 RELr426 190762
# Run by YUKI on Sun Jun 23 06:06:39 2024
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.32292, 1.32407), width=194.733, 
    height=131.348)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('abaqus/braiding.py', __main__.__dict__)
#: A new model database has been created.
#: The model "Model-1" has been created.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
#: The interaction property "IntProp-1" has been created.
#: The interaction "Int-1" has been created.
#: CS  [-75.0, -75.0, 0.0]
#: CS  [-37.5, -75.0, 0.0]
#: CS  [0.0, -75.0, 0.0]
#: CS  [37.5, -75.0, 0.0]
#: CS  [75.0, -75.0, 0.0]
#: CS  [-75.0, -37.5, 0.0]
#: CS  [-37.5, -37.5, 0.0]
#: CS  [0.0, -37.5, 0.0]
#: CS  [37.5, -37.5, 0.0]
#: CS  [75.0, -37.5, 0.0]
#: CS  [-75.0, 0.0, 0.0]
#: CS  [-37.5, 0.0, 0.0]
#: CS  [0.0, 0.0, 0.0]
#: CS  [37.5, 0.0, 0.0]
#: CS  [75.0, 0.0, 0.0]
#: CS  [-75.0, 37.5, 0.0]
#: CS  [-37.5, 37.5, 0.0]
#: CS  [0.0, 37.5, 0.0]
#: CS  [37.5, 37.5, 0.0]
#: CS  [75.0, 37.5, 0.0]
#: CS  [-75.0, 75.0, 0.0]
#: CS  [-37.5, 75.0, 0.0]
#: CS  [0.0, 75.0, 0.0]
#: CS  [37.5, 75.0, 0.0]
#: CS  [75.0, 75.0, 0.0]
print('RT script done')
#: RT script done
