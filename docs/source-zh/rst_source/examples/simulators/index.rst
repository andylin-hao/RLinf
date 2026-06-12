基于模拟器的具身强化学习
========================

本类示例以 **模拟器（基准）** 为主线，展示如何在某个仿真平台上运行 RLinf —— 包括环境安装、资产路径、观测/动作空间，以及一个参考 RL 训练配方（通常为 PPO 或 GRPO + VLA 策略）。

如果你的出发点是 "我想在基准 *X* 上训练"，那这里就是合适的入口。若以模型为主线（pi₀、GR00T 等）请参考 :doc:`../vla_wam/index`，真机部署请参考 :doc:`../real_world/index`。

LIBERO 系列
-----------

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/libero_numbers.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/libero.html" style="text-decoration: underline; color: blue;">
           <b>基于LIBERO的强化学习</b>
         </a><br>
         LIBERO+OpenVLA-OFT+GRPO成功率达99%
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/378920588652fff0a2a0b163b392c94694993345/pic/libero-plus.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/liberoplus_pro.html" style="text-decoration: underline; color: blue;">
           <b>基于 LIBERO-Pro 与 LIBERO-Plus 的强化学习</b>
         </a><br>
         支持 LIBERO-Pro / LIBERO-Plus + OpenVLA-OFT / π₀ / π₀.₅ + PPO/GRPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/libero_numbers.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/libero_amd.html" style="text-decoration: underline; color: blue;">
           <b>AMD ROCm 平台上的 LIBERO 强化学习</b>
         </a><br>
         LIBERO 强化学习的 ROCm 依赖安装与 OSMesa 渲染配置
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/libero_numbers.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/libero_ascend.html" style="text-decoration: underline; color: blue;">
           <b>Ascend CANN 平台上的 LIBERO 强化学习</b>
         </a><br>
         LIBERO 强化学习的 CANN 依赖安装与驱动挂载配置
       </p>
     </div>

   </div>

其他模拟器
----------

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <video controls autoplay loop muted playsinline preload="metadata" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
         <source src="https://github.com/RLinf/misc/raw/main/pic/embody.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/maniskill.html" style="text-decoration: underline; color: blue;">
           <b>基于ManiSkill的强化学习</b>
         </a><br>
         ManiSkill+OpenVLA+PPO/GRPO达到SOTA训练效果
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/behavior.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/behavior.html" style="text-decoration: underline; color: blue;">
           <b>基于Behavior的强化学习</b>
         </a><br>
         支持Behavior+OpenVLA-OFT+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/metaworld.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/metaworld.html" style="text-decoration: underline; color: blue;">
           <b>基于MetaWorld的强化学习</b>
         </a><br>
         支持MetaWorld+π₀/π₀.₅+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/IsaacLab.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/isaaclab.html" style="text-decoration: underline; color: blue;">
           <b>基于IsaacLab的强化学习</b>
         </a><br>
         支持IsaacLab+gr00t+PPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/calvin.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/calvin.html" style="text-decoration: underline; color: blue;">
           <b>基于CALVIN的强化学习</b>
         </a><br>
         支持CALVIN+π₀/π₀.₅+PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/robocasa.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/robocasa.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboCasa的强化学习</b>
         </a><br>
         支持RoboCasa+π₀+GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/assets/files/50_tasks.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/robotwin.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboTwin的强化学习</b>
         </a><br>
         支持RoboTwin + OpenVLA-OFT/π₀/π₀.₅ + PPO/GRPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RoboVerseOrg/RoboVerse/main/docs/source/metasim/images/tea.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/roboverse.html" style="text-decoration: underline; color: blue;">
           <b>基于RoboVerse的强化学习</b>
         </a><br>
         支持RoboVerse + π₀.₅ + PPO训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/serl/refs/heads/RLinf/franka-sim/franka_sim/franka_sim/envs/xmls/robotiq_2f85/2f85.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);"
            data-target="animated-image.originalImage">
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/frankasim.html" style="text-decoration: underline; color: blue;">
           <b>基于Franka-Sim的强化学习</b>
         </a><br>
         支持Franka-Sim+MLP/CNN+PPO/SAC训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/embodichain.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/embodichain.html" style="text-decoration: underline; color: blue;">
           <b>基于 EmbodiChain 的强化学习</b>
         </a><br>
         使用 EmbodiChain gym 任务进行 MLP + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/polaris.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/polaris.html" style="text-decoration: underline; color: blue;">
           <b>基于 PolaRiS 仿真平台的强化学习</b>
         </a><br>
         PolaRiS + OpenPI + PPO 训练桌面操作任务
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gsenv.gif"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/gsenv.html" style="text-decoration: underline; color: blue;">
           <b>基于 GSEnv 的 Real2Sim2Real 强化学习</b>
         </a><br>
         支持 GSEnv + π₀.₅ + PPO 训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="../embodied/genesis.html" style="text-decoration: underline; color: blue;">
           <b>基于 Genesis 的强化学习</b>
         </a><br>
         在 Genesis 仿真平台上进行 MLP 策略训练
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   ../embodied/maniskill
   ../embodied/libero
   ../embodied/libero_amd
   ../embodied/libero_ascend
   ../embodied/liberoplus_pro
   ../embodied/behavior
   ../embodied/metaworld
   ../embodied/isaaclab
   ../embodied/calvin
   ../embodied/robocasa
   ../embodied/robotwin
   ../embodied/roboverse
   ../embodied/frankasim
   ../embodied/embodichain
   ../embodied/polaris
   ../embodied/gsenv
   ../embodied/genesis
