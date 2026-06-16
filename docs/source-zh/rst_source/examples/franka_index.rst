Franka 机器人
====================

当你的真机流程从 Franka 机械臂开始时，请使用本节。这里集中放置基础 RL 配方、reward model 训练、ZED 与 Robotiq 硬件、GELLO 遥操作、双臂 Franka、灵巧手、Pi0 SFT 和 HG-DAgger 在线干预训练。

先阅读 **Real-World RL**，完成通用 Ray、Franka 与安全流程；然后根据实验室硬件选择对应的训练或部署变体。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka.html" style="text-decoration: underline; color: blue;">
           <b>Real-World RL</b>
         </a><br>
         在真实 Franka 机械臂上训练和评测 RL 策略
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_reward_model.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_reward_model.html" style="text-decoration: underline; color: blue;">
           <b>Reward Model</b>
         </a><br>
         使用学习得到的 reward model 辅助 Franka 操作任务
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/robotiq_zed.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_zed_robotiq.html" style="text-decoration: underline; color: blue;">
           <b>ZED + Robotiq</b>
         </a><br>
         配置 ZED 相机、Robotiq 夹爪和数据采集流程
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gello.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_gello.html" style="text-decoration: underline; color: blue;">
           <b>GELLO</b>
         </a><br>
         安装、配置并验证 Franka 的 GELLO 遥操作流程
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dual_franka.html" style="text-decoration: underline; color: blue;">
           <b>Dual-Arm</b>
         </a><br>
         运行双 Franka 的 GELLO 采集、π₀.₅ SFT 和部署流程
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dexhand.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_dexhand.html" style="text-decoration: underline; color: blue;">
           <b>Dexterous Hand</b>
         </a><br>
         组合 Franka 机械臂与睿研五指灵巧手
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_pi0_sft_deploy.html" style="text-decoration: underline; color: blue;">
           <b>Pi0 SFT</b>
         </a><br>
         采集数据、微调 Pi0，并将 checkpoint 部署到 Franka
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/hg-dagger.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/hg-dagger.html" style="text-decoration: underline; color: blue;">
           <b>HG-DAgger</b>
         </a><br>
         运行 Human-Gated DAgger 的采集、SFT 和在线干预训练流程
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
