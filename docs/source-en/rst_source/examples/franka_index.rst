Franka Robots
=============

Use this section when your real-world workflow starts from a Franka arm. It groups the base RL recipe with reward-model training, ZED and Robotiq hardware, GELLO teleoperation, dual-arm rigs, dexterous hands, Pi0 SFT, and HG-DAgger intervention training.

Start with **Real-World RL** for the common Ray, Franka, and safety workflow. Then choose the hardware or training variant that matches your lab setup.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka.html" style="text-decoration: underline; color: blue;">
           <b>Real-World RL</b>
         </a><br>
         Train and evaluate RL policies on a physical Franka arm
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_reward_model.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_reward_model.html" style="text-decoration: underline; color: blue;">
           <b>Reward Model</b>
         </a><br>
         Use a learned reward model to guide Franka manipulation
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/robotiq_zed.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_zed_robotiq.html" style="text-decoration: underline; color: blue;">
           <b>ZED + Robotiq</b>
         </a><br>
         Set up ZED cameras, a Robotiq gripper, and data collection
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gello.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_gello.html" style="text-decoration: underline; color: blue;">
           <b>GELLO</b>
         </a><br>
         Install, configure, and verify GELLO teleoperation for Franka
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dual_franka.html" style="text-decoration: underline; color: blue;">
           <b>Dual-Arm</b>
         </a><br>
         Run the dual-Franka GELLO collection, π₀.₅ SFT, and deployment flow
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dexhand.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_dexhand.html" style="text-decoration: underline; color: blue;">
           <b>Dexterous Hand</b>
         </a><br>
         Combine a Franka arm with a Ruiyan five-finger dexterous hand
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_pi0_sft_deploy.html" style="text-decoration: underline; color: blue;">
           <b>Pi0 SFT</b>
         </a><br>
         Collect data, fine-tune Pi0, and deploy the checkpoint on Franka
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/hg-dagger.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/hg-dagger.html" style="text-decoration: underline; color: blue;">
           <b>HG-DAgger</b>
         </a><br>
         Run human-gated DAgger with collection, SFT, and online intervention
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   Real-World RL <embodied/franka>
   Reward Model <embodied/franka_reward_model>
   ZED + Robotiq <embodied/franka_zed_robotiq>
   GELLO <embodied/franka_gello>
   Dual-Arm <embodied/dual_franka>
   Dexterous Hand <embodied/franka_dexhand>
   Pi0 SFT <embodied/franka_pi0_sft_deploy>
   HG-DAgger <embodied/hg-dagger>
