On Ascend 950 NPUs, add ``--torch 2.11.0`` before ``embodied`` in the native
installation command. The Ascend installer uses PyTorch and ``torch-npu`` 2.6.0
by default, and ``torch-npu`` 2.6.0 fails to initialize a 950 NPU with
``Unsupported soc version``. PyTorch and ``torch-npu`` 2.11.0 have been verified
on Ascend 950 with NPU driver 25.7.rc1 and CANN 9.1.1.

.. warning::

   Keep the default PyTorch version on other Ascend NPUs. RLinf passes weights
   between collocated workers through NPU IPC. On the Ascend 910B hosts used for
   CI, ``torch-npu`` 2.10 through 2.12 cannot use NPU IPC with the installed
   driver: weight sync fails with ``entry in cache has missing shared_ptr`` and
   the run stops making progress. Select ``--torch 2.11.0`` on those hosts only
   after a collocated run completes its first training step.
