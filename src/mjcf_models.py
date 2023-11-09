

MJCF_single_pend_motor = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <material name="grid" texture="grid" texrepeat="1 1"
     texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <geom name="floor" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid"/>
    <site name="anchor" pos="0 0 .3" size=".01"/>
    <camera name="fixed" pos="0 -2.5 1" xyaxes="1 0 0 0 0 1"/>

    
    <geom name="pole" type="cylinder" fromto="0 1 1 0 0 1" size=".02"/>

    <body name="pend1" pos="0 0 1">
      <joint name="swing1" type="hinge" damping="0.1" axis="0 -1 0"/>
      <geom name="pend1" type="capsule" fromto="0 0 0 0 0 -0.4" size=".04" rgba="0 0 1 1"/>
    </body> 
  </worldbody>

  <actuator>
    <motor name="my_motor" joint="swing1" gear="1"/>
  </actuator>

</mujoco>
"""


MJCF_dpend_outer_motor = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <material name="grid" texture="grid" texrepeat="1 1"
     texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <geom name="floor" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid"/>
    <site name="anchor" pos="0 0 .3" size=".01"/>
    <camera name="fixed" pos="0 -2.5 1" xyaxes="1 0 0 0 0 1"/>

    
    <geom name="pole" type="cylinder" fromto="0 1 1 0 0 1" size=".02"/>

    <body name="pend1" pos="0 0 1">
      <joint name="swing1" type="hinge" damping="0.1" axis="0 -1 0"/>
      <geom name="pend1" type="capsule" fromto="0 0 0 0 0 -0.4" size=".04" rgba="0 0 1 1"/>
      <body name="pend2" pos="0 -0.041 -0.4">
        <joint name="swing2" type="hinge" damping="0.1" axis="0 1 0"/>
        <geom name="pend2" type="capsule" fromto="0 0 0 0 0 -0.4" size=".04" rgba="0 0 1 1"/>
      </body>
    </body> 
  </worldbody>

  <actuator>
    <motor name="my_motor" joint="swing2" gear="1"/>
  </actuator>

</mujoco>
"""

MJCF_single_pend_2 = """
<mujoco>
<option gravity="0 0 -9.81">
    <flag sensornoise="enable"/>
</option>

<worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <camera name="fixed" pos="0 -2.5 2" xyaxes="1 0 0 0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 2" euler="0 180 0">
        <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 0.5"/>
        <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>
    </body>
</worldbody>

<actuator>
    <motor joint="pin" name="torque" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
</actuator>

</mujoco>
                  """

MJCF_single_pend_w_sensor = """
<mujoco>
<option gravity="0 0 -9.81">
    <flag sensornoise="enable"/>
</option>

<worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <camera name="fixed" pos="0 -2.5 2" xyaxes="1 0 0 0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 2" euler="0 180 0">
        <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 0.5"/>
        <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>
    </body>
</worldbody>

<actuator>
    <motor joint="pin" name="torque" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
    <position name="position_servo" joint="pin" kp="10"/>
    <velocity name="velocity_servo" joint="pin" kv="0"/>
</actuator>

<sensor>
    <jointpos joint="pin" noise="0.2"/>
    <jointvel joint="pin" noise="1"/>
</sensor>
</mujoco>
                  """