.. role:: hidden
    :class: hidden-section

=========
distdl.nn
=========

.. toctree::
    :glob:
    :hidden:

    nn/*

Overview
========

These are the public interfaces.


Distributed Containers
======================

.. list-table::
   :widths: 35 65
   :width: 100%
   :header-rows: 0
   :align: left

   * - :ref:`code_reference/nn/module:Base Distributed Module`
     - Base class of all distributed layers in DistDL.

Primitive Distributed Data Movement Layers
==========================================

We implement a number of primitives.

.. list-table::
   :widths: 35 65
   :width: 100%
   :header-rows: 0
   :align: left

   * - :ref:`code_reference/nn/broadcast:Broadcast Layer`
     - Performs a broadcast of a tensor from one partition to another.
   * - :ref:`code_reference/nn/halo_exchange:Halo Exchange Layer`
     - Performs the halo exchange operation of a tensor on a partition.
   * - :ref:`code_reference/nn/sum_reduce:SumReduce Layer`
     - Performs a sum-reduction of a tensor from one partition to another.
   * - :ref:`code_reference/nn/transpose:Transpose Layer`
     - Performs a transpose or a tensor from one partition to another.

Distributed Comptutation Layers
===============================

We implement a number of distributed layers based on the actual layers and the primitives.

.. list-table::
   :widths: 35 65
   :width: 100%
   :header-rows: 0
   :align: left

   * - :ref:`code_reference/nn/convolution:Convolution Layers`
     - Distributed convolutional layers.
   * - :ref:`code_reference/nn/pooling:Pooling Layers`
     - Distributed pooling layers.
   * - :ref:`code_reference/nn/linear:Linear Layer`
     - Distributed linear layer.
   * - :ref:`code_reference/nn/upsampling:Upsample Layer`
     - Distributed upsampling layer.

Additional Sequential Layers
============================

We implement some useful sequential modules.

.. list-table::
   :widths: 35 65
   :width: 100%
   :header-rows: 0
   :align: left

   * - :ref:`code_reference/nn/interpolate:Interpolate Layer`
     - N-dimensional, interpolation.

   * - :ref:`code_reference/nn/padding:Padding Layer`
     - N-dimensional, unbalanced padding.

   * - :ref:`code_reference/nn/unpadding:Unpadding Layer`
     - N-dimensional, unbalanced unpadding.
