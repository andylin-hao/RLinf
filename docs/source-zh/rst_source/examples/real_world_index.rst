真实机器人强化学习
====================

当你的出发点是 Franka 系列之外的真实机器人硬件时，请使用本节。这里的配方覆盖 GimArm、XSquare Turtle2 和 Dexmal DOS-W1 的遥操作、数据采集、Sim-to-Real 迁移和在线 RL 流程。

Franka 机械臂及其变体请进入专门的 :doc:`franka_index` 章节。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/gim-arm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gim_arm.html" style="text-decoration: underline; color: blue;">
           <b>GimArm</b>
         </a><br>
         在 GimArm 六自由度机械臂上通过 SocketCAN 与 Pinocchio FK 训练 peg-insertion 任务
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/xsquare_turtle2.html" style="text-decoration: underline; color: blue;">
           <b>XSquare Turtle2</b>
         </a><br>
         在 XSquare Turtle2 双臂机器人上运行 SAC + CNN 策略
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dosw1.html" style="text-decoration: underline; color: blue;">
           <b>Dexmal DOS-W1</b>
         </a><br>
         在 Dexmal DOS-W1 双臂机器人上训练 Flow Matching + SAC 抓取任务
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   GimArm <embodied/gim_arm>
   XSquare Turtle2 <embodied/xsquare_turtle2>
   DOS-W1 <embodied/dosw1>
