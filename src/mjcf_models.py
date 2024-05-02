

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
      <geom name="pend1" type="capsule" fromto="0 0 0 0 0 -0.4" size=".04" rgba="0 1 0 1"/>
      <body name="pend2" pos="0 -0.041 -0.4">
        <joint name="swing2" type="hinge" damping="0.1" axis="0 1 0"/>
        <geom name="pend2" type="capsule" fromto="0 0 0 0 0 -0.4" size=".04" rgba="1 0 0 1"/>
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
    <body name="body1" pos="0 0 2" euler="0 180 0">
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

def create_MJCF_single_pend_m_d_mod(mass, damping, length):
    mjcf_pend_model = """
                      <mujoco>
                      <option gravity="0 0 -9.81">
                          <flag sensornoise="enable"/>
                      </option>

                      <worldbody>
                          <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                          <camera name="fixed" pos="0 -5 1" xyaxes="1 0 0 0 0 1"/>
                          <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                          <body name="pend1" pos="0 0 2" euler="0 0 0">
                              <joint name="pin" type="hinge" axis="0 -1 0" pos="0 0 0.5"/>
                              <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>
                          </body>
                      </worldbody>

                      <actuator>
                          <motor joint="pin" name="torque" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
                      </actuator>

                      </mujoco>
                                    """
    return mjcf_pend_model

                                      # <inertial pos="0 0 -1.5" mass="1.0" diaginertia="0.5 1.0 0.5"/>
def create_MJCF_double_pend_m_d_mod(mass, damping, length):
    mjcf_pend_model = """<mujoco>
                          <option gravity="0 0 -9.81">
                              <flag sensornoise="disable"/>
                          </option>

                          <worldbody>
                              <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                              <camera name="fixed" pos="0 -4 2" xyaxes="1 0 0 0 0 1"/>
                              <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                              <body name="pend1" pos="0 0 3" euler="0 0 0">
                                  <joint name="swing1" type="hinge" axis="0 -1 0" pos="0 0 0"/>
                                  <geom type="cylinder" mass="1.0"fromto="0 0 0 0 0 -1.0" size=".05" rgba="0 1 0 1"/>
                                  <body name="pend2" pos="0 0.0 -1.0"  euler="0 0 0">
                                      <joint name="swing2" type="hinge" damping="0.0" axis="0 -1 0" pos="0 0 0"/>
                                      <geom type="cylinder" mass="1.0" fromto="0 0 0 0 0 -1.0" size=".05" rgba="0 1 0 1"/> 
                                  </body>                              
                              </body>
                          </worldbody>

                          <contact>
                              <exclude body1="pend1" body2="pend2" />
                          </contact>

                          <actuator>
                              <motor joint="swing1" name="torque1" gear="1" ctrllimited="true" ctrlrange="-1000 1000"/>
                              <motor joint="swing2" name="torque2" gear="1" ctrllimited="true" ctrlrange="-1000 1000"/>                              
                          </actuator>

                          </mujoco>
                                        """
    return mjcf_pend_model


                                  # <body name="pend2" pos="0 0 0.5"  euler="0 0 0">
                                  #     <joint name="swing2" type="hinge" damping="0.1" axis="0 -1 0" pos="0 0 0.5"/>
                                  #     <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" mass="1"/>



  #   <body name="pend1" pos="0 0 1">
  #     <joint name="swing1" type="hinge" damping="0.1" axis="0 -1 0"/>
  #     <geom name="pend1" type="capsule" fromto="0 0 0 0 0 -0.4" size=".04" rgba="0 0 1 1"/>
  #     <body name="pend2" pos="0 -0.041 -0.4">
  #       <joint name="swing2" type="hinge" damping="0.1" axis="0 1 0"/>
  #       <geom name="pend2" type="capsule" fromto="0 0 0 0 0 -0.4" size=".04" rgba="0 0 1 1"/>
  #     </body>
  #   </body> 
  # </worldbody>

inverted_cartpole = """ <mujoco model="inverted pendulum">
	<compiler inertiafromgeom="true"/>
	<default>
		<joint armature="0" damping="1" limited="true"/>
		<geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
		<tendon/>
		<motor ctrlrange="-3 3"/>
	</default>
	<option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
	<size nstack="3000"/>
	<worldbody>
		<!--geom name="ground" type="plane" pos="0 0 0" /-->
		<geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
		<body name="cart" pos="0 0 0">
			<joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
			<geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
			<body name="pole" pos="0 0 0">
				<joint axis="0 1 0" name="hinge" pos="0 0 0" range="-90 90" type="hinge"/>
				<geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
				<!--                 <body name="pole2" pos="0.001 0 0.6"><joint name="hinge2" type="hinge" pos="0 0 0" axis="0 1 0"/><geom name="cpole2" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05 0.3" rgba="0.7 0 0.7 1"/><site name="tip2" pos="0 0 .6"/></body>-->
			</body>
		</body>
	</worldbody>
	<actuator>
		<motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
	</actuator>
</mujoco>
"""


def create_MJCF_double_pend_DM():
  '''https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/acrobot.xml'''
  double_pend_model = '''
    <mujoco model="acrobot">
      <!--
      <include file="./common/visual.xml"/>
      <include file="./common/skybox.xml"/>
      <include file="./common/materials.xml"/>
      -->

      <default>
        <joint damping=".05"/>
        <geom type="capsule" mass="1"/>
      </default>

      <option timestep="0.01" integrator="RK4">
        <flag constraint="disable" energy="enable"/>
      </option>

      <worldbody>
        <light name="light" pos="0 0 6"/>
        <geom name="floor" size="3 3 .2" type="plane"/>
        <site name="target" type="sphere" pos="0 0 4" size="0.2" group="3"/>
        <camera name="fixed" pos="0 -6 2" zaxis="0 -1 0"/>
        <camera name="lookat" mode="targetbodycom" target="upper_arm" pos="0 -2 3"/>
        <body name="upper_arm" pos="0 0 2">
          <joint name="shoulder" type="hinge" axis="0 1 0"/>
          <geom name="upper_arm_decoration"  type="cylinder" fromto="0 -.06 0 0 .06 0" size="0.051" mass="0"/>
          <geom name="upper_arm" fromto="0 0 0 0 0 1" size="0.05" />
          <body name="lower_arm" pos="0 0 1">
            <joint name="elbow" type="hinge" axis="0 1 0"/>
            <geom name="lower_arm" fromto="0 0 0 0 0 1" size="0.049" />
            <site name="tip" pos="0 0 1" size="0.01"/>
          </body>
        </body>
      </worldbody>

      <actuator>
        <motor name="elbow" joint="elbow" gear="2" ctrllimited="true" ctrlrange="-100 100"/>
      </actuator>
    </mujoco>
    '''
  return double_pend_model


def create_MJCF_double_cartpole_gym():
    '''https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/inverted_double_pendulum.xml'''
    double_pend_model = '''
<mujoco model="cartpole">
  <compiler coordinate="local" inertiafromgeom="true"/>
  <custom>
    <numeric data="2" name="frame_skip"/>
  </custom>
  <default>
    <joint damping="0.05"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="1e-5 0 -9.81" integrator="RK4" timestep="0.01"/>
  <size nstack="3000"/>
  <worldbody>
    <geom name="floor" pos="0 0 -3.0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
    <body name="cart" pos="0 0 0">
      <joint axis="1 0 0" limited="true" margin="0.01" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
      <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
      <body name="pole" pos="0 0 0">
        <joint axis="0 1 0" name="hinge" pos="0 0 0" type="hinge"/>
        <geom fromto="0 0 0 0 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
        <body name="pole2" pos="0 0 0.6">
          <joint axis="0 1 0" name="hinge2" pos="0 0 0" type="hinge"/>
          <geom fromto="0 0 0 0 0 0.6" name="cpole2" rgba="0 0.7 0.7 1" size="0.045 0.3" type="capsule"/>
          <site name="tip" pos="0 0 .6" size="0.01 0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1000 1000" gear="1" joint="slider" name="slide"/>
  </actuator>
</mujoco>
      '''
    return double_pend_model


def create_acrobot():
    acrobot_model = '''
<mujoco>
  <option timestep="0.05" integrator="RK4">
    <flag sensornoise="enable" contact="disable" energy="enable"/>
  </option>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <camera name="fixed" pos="0 -4 2" xyaxes="1 0 0 0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 2.5" euler="0 180 0">
      <joint name="shoulder" type="hinge" pos="0 0 0.5" axis="0 -1 0" />
      <geom type="cylinder" size=".05 .5" rgba="0 .9 0 1" />
      <inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      
        <body pos="0 0 -1" euler="0 0 0">
          <joint name="elbow" type="hinge" pos="0 0 0.5" axis="0 -1 0" />
          <geom type="cylinder" size=".05 .5" rgba="0 0 0.9 1" />
          <inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
        </body>

    </body>
  </worldbody>

  <actuator>
    <motor name="torque_elbow" joint="elbow" gear="1" ctrlrange="-1000 1000" ctrllimited="true"/>
  </actuator>
</mujoco>
      '''
    return acrobot_model
