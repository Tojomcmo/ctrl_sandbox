
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


def create_MJCF_pm_pend_dev():
    mjcf_pend_model ="""<mujoco>
                          <option gravity="0 0 -9.81">
                              <flag sensornoise="disable"/>
                          </option>

                          <worldbody>
                              <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                              <camera name="fixed" pos="0 -4 2" xyaxes="1 0 0 0 0 1"/>
                                <body name="pendulum1" pos="0 0 0">
                                    <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0"/>
                                    <geom type="cylinder" size="0.05 0.5" pos="0 0 -1.0" rgba="0.8 0.1 0.1 1"/>
                                    <inertial pos="0 0 -1.0" mass="1" diaginertia="0.0001 0.0001 0.00001"/>/>
                                </body>   
                          </worldbody>

                          <actuator>
                              <motor joint="joint1" name="shoulder_motor" gear="1" ctrllimited="false"/>                    
                          </actuator>

                          </mujoco>
                                        """
    return mjcf_pend_model


def create_MJCF_single_pend_dev():
    mjcf_pend_model ="""<mujoco>
                          <option gravity="0 0 -9.81">
                              <flag sensornoise="disable"/>
                          </option>

                          <worldbody>
                              <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                              <camera name="fixed" pos="0 -4 2" xyaxes="1 0 0 0 0 1"/>
                                <body name="pendulum1" pos="0 0 0">
                                    <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0"/>
                                    <geom type="cylinder" size="0.05 0.5" pos="0 0 -0.5" rgba="0.8 0.1 0.1 1"/>
                                    <inertial pos="0 0 -0.5" mass="1" diaginertia="0.084 0.084 0.00125"/>/>
                                </body>   
                          </worldbody>

                          <actuator>
                              <motor joint="joint1" name="shoulder_motor" gear="1" ctrllimited="false"/>                    
                          </actuator>

                          </mujoco>
                                        """
    return mjcf_pend_model

                            #   <body name="base" pos="0 0 3">
                            #     <!-- First pendulum arm -->

def create_MJCF_double_pend_dev():
    mjcf_pend_model = """<mujoco>
                          <option gravity="0 0 -9.81">
                              <flag sensornoise="disable"/>
                          </option>

                          <worldbody>
                              <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                              <camera name="fixed" pos="0 -4 2" xyaxes="1 0 0 0 0 1"/>
                              <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                              <body name="base" pos="0 0 3">
                                <!-- First pendulum arm -->
                                <body name="pendulum1" pos="0 0 0">
                                    <!-- Hinge joint at the origin of the base, rotation about y-axis -->
                                    <joint name="joint1" type="hinge" axis="0 1 0" pos="0 0 0"/>
                                    <!-- Cylinder geometry with length 1 meter (2*0.5), centered along local z-axis -->
                                    <geom type="cylinder" size="0.05 0.5" pos="0 0 -0.5" rgba="0.8 0.1 0.1 1"/>
                                    <!-- Center of mass at the center of the cylinder -->
                                    <inertial pos="0 0 -0.5" mass="1" diaginertia="0.084 0.084 0.00125"/>/>
                                    <!-- Second pendulum arm, positioned to hang from the end of the first arm -->
                                    <body name="pendulum2" pos="0 0 -1">
                                      <!-- Hinge joint at the connection point with the first arm -->
                                      <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0"/>
                                      <!-- Cylinder geometry with length 1 meter, centered along local z-axis -->
                                      <geom type="cylinder" size="0.05 0.5" pos="0 0 -0.5" rgba="0.1 0.8 0.1 1"/>
                                      <!-- Center of mass at the center of the cylinder -->
                                      <inertial pos="0 0 -0.5" mass="1" diaginertia="0.084 0.084 0.00125"/>/>
                                    </body>
                                </body>
                              </body>    
                          </worldbody>

                          <contact>
                              <exclude body1="pendulum1" body2="pendulum2" />
                          </contact>

                          <actuator>
                              <motor joint="joint1" name="torque1" gear="1" ctrllimited="true" ctrlrange="-1000 1000"/>                    
                          </actuator>

                          </mujoco>
                                        """
    return mjcf_pend_model


     # <motor joint="joint2" name="torque2" gear="1" ctrllimited="true" ctrlrange="-1000 1000"/>    
