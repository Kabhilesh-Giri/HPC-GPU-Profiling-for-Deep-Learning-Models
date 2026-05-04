# Real-Time High-Performance Computing

<h3 align="center">GPU Performance Monitoring for Deep Learning Training Models</h3>

<p align="center">
  GPU utilization, memory pressure, power draw, temperature, clock speed, and throughput analysis for
  <b>DeepFilterNet</b> and <b>Wave-U-Net</b> training on NVIDIA Tesla P100 and V100 GPUs.
</p>

<p align="center">
  <b>Authors:</b> Jashwanth Meganathan and Kabhilesh Giri
</p>

<p align="center">
  <a href="EECE5640_ProjectPresentation_Jashwanth%26Kabhilesh.pptx">Project Presentation</a>
  &nbsp;|&nbsp;
  <a href="EECE_5640_Jashwanth%26Kabhilesh.pdf">Project PDF</a>
</p>

<p align="center">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-P100%20%7C%20V100-0b7285">
  <img alt="Models" src="https://img.shields.io/badge/Models-DeepFilterNet%20%7C%20Wave--U--Net-2f9e44">
  <img alt="Focus" src="https://img.shields.io/badge/Focus-HPC%20Monitoring-f08c00">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.x-6741d9">
</p>

## Project Snapshot

This project evaluates how two audio deep learning models behave under GPU training workloads:

| Model | Task | Why it matters |
| --- | --- | --- |
| DeepFilterNet | Speech enhancement and noise suppression | Useful for low-latency audio cleanup and ANC-style edge AI workloads |
| Wave-U-Net | End-to-end waveform audio separation | Trains directly on raw audio waveforms and stresses memory, clocks, and sustained utilization |

The experiments compare NVIDIA Tesla P100 and V100 GPUs using GPU telemetry collected during training.

## Hardware and Metrics

| Resource | Tesla P100 | Tesla V100 |
| --- | ---: | ---: |
| Form factor | PCIe | SXM2 |
| GPU memory | 12 GB | 32 GB |
| CUDA version used | 12.2 | 12.3 |

Metrics monitored:

| Category | Measurements |
| --- | --- |
| Throughput | Samples processed per second |
| Utilization | GPU utilization and memory controller utilization |
| Memory | Total, used, free, and utilization percentage |
| Power | Power draw, limit, and management status |
| Thermal | GPU temperature and stability |
| Clocks | SM clock and memory clock behavior |
| Performance state | P-state during training |

## Key Results

### DeepFilterNet

| Observation | Result |
| --- | --- |
| V100 vs P100 | V100 delivered stronger throughput, higher usable GPU utilization, and more stable scaling. |
| Batch size behavior | Batch size 1 was stable on both GPUs. Batch size 2 improved V100 efficiency but exposed P100 limitations. |
| Dataset pressure | MS-SNSD was more demanding than MS-DNS, especially at larger batch sizes. |
| Best fit | V100 is better for production-scale DFN training; P100 is better for lightweight edge-style simulation. |

<table>
  <tr>
    <td width="50%">
      <img src="assets/readme/dfn_dns_training.png" alt="DeepFilterNet DNS training curves">
      <p align="center"><b>DeepFilterNet training on MS-DNS</b></p>
    </td>
    <td width="50%">
      <img src="assets/readme/dfn_snsd_training.png" alt="DeepFilterNet SNSD training curves">
      <p align="center"><b>DeepFilterNet training on MS-SNSD</b></p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%">
      <img src="assets/readme/dfn_dns_p100_utilization.png" alt="P100 utilization on DNS">
      <p align="center"><b>P100 utilization on DNS</b></p>
    </td>
    <td width="50%">
      <img src="assets/readme/dfn_dns_v100_utilization.png" alt="V100 utilization on DNS">
      <p align="center"><b>V100 utilization on DNS</b></p>
    </td>
  </tr>
</table>

### Wave-U-Net

Wave-U-Net was evaluated using batch sizes 1 and 4. The P100 maintained high utilization with lower power draw, while the V100 provided higher clocks and more memory headroom but consumed substantially more power at batch size 4.

| GPU | Batch | Temperature | Power | Memory Utilization | Memory Used | GPU Utilization | SM Clock |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| P100 | 1 | 52 to 54 C, stable | 100 to 130 W | 22 to 26% | 1900 MiB | 90 to 95% | 1300 MHz, stable |
| P100 | 4 | 63 to 64 C, stable | 140 to 170 W | 26 to 27% | 3600 MiB | 92 to 96% | 1300 MHz, stable |
| V100 | 1 | 63 to 64 C, stable | 170 to 190 W | 35 to 37% | 1900 MiB | 85 to 90% | 1530 MHz, slightly lower |
| V100 | 4 | 64 to 66 C, fluctuating | 250 to 300 W | 44 to 46% | 3400 MiB | 92 to 95% | 1530 MHz, stable at max |

<table>
  <tr>
    <td width="50%">
      <img src="assets/readme/waveunet_p100_gpu_utilization.png" alt="Wave-U-Net P100 GPU utilization">
      <p align="center"><b>P100 GPU utilization</b></p>
    </td>
    <td width="50%">
      <img src="assets/readme/waveunet_v100_gpu_utilization.png" alt="Wave-U-Net V100 GPU utilization">
      <p align="center"><b>V100 GPU utilization</b></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="assets/readme/waveunet_p100_power.png" alt="Wave-U-Net P100 power draw">
      <p align="center"><b>P100 power draw</b></p>
    </td>
    <td width="50%">
      <img src="assets/readme/waveunet_v100_power.png" alt="Wave-U-Net V100 power draw">
      <p align="center"><b>V100 power draw</b></p>
    </td>
  </tr>
</table>

## Result Gallery

<details>
<summary><b>DeepFilterNet throughput plots</b></summary>

<table>
  <tr>
    <td><img src="assets/readme/dfn_throughput_snsd_p100.png" alt="DFN SNSD P100 throughput"></td>
    <td><img src="assets/readme/dfn_throughput_snsd_v100.png" alt="DFN SNSD V100 throughput"></td>
  </tr>
  <tr>
    <td><img src="assets/readme/dfn_throughput_dns_p100.png" alt="DFN DNS P100 throughput"></td>
    <td><img src="assets/readme/dfn_throughput_dns_v100.png" alt="DFN DNS V100 throughput"></td>
  </tr>
</table>

</details>

<details>
<summary><b>Wave-U-Net V100 monitoring plots</b></summary>

<table>
  <tr>
    <td><img src="assets/readme/waveunet_v100_temperature.png" alt="V100 temperature"></td>
    <td><img src="assets/readme/waveunet_v100_memory_utilization.png" alt="V100 memory utilization"></td>
  </tr>
  <tr>
    <td><img src="assets/readme/waveunet_v100_memory_usage.png" alt="V100 memory usage"></td>
    <td><img src="assets/readme/waveunet_v100_sm_clock.png" alt="V100 SM clock"></td>
  </tr>
</table>

</details>

<details>
<summary><b>Wave-U-Net P100 monitoring plots</b></summary>

<table>
  <tr>
    <td><img src="assets/readme/waveunet_p100_temperature.png" alt="P100 temperature"></td>
    <td><img src="assets/readme/waveunet_p100_memory_utilization.png" alt="P100 memory utilization"></td>
  </tr>
  <tr>
    <td><img src="assets/readme/waveunet_p100_memory_usage.png" alt="P100 memory usage"></td>
    <td><img src="assets/readme/waveunet_p100_sm_clock.png" alt="P100 SM clock"></td>
  </tr>
</table>

</details>

## Repository Contents

| Path | Description |
| --- | --- |
| `Wave-U-Net-Pytorch/` | Wave-U-Net PyTorch implementation used for training analysis |
| [`gpu_metrics_log_P100_Batch_Size_1.csv`](gpu_metrics_log_P100_Batch_Size_1.csv) | P100 GPU telemetry log for batch size 1 |
| [`gpu_metrics_log_P100_Batch_Size_4.csv`](gpu_metrics_log_P100_Batch_Size_4.csv) | P100 GPU telemetry log for batch size 4 |
| [`EECE5640_ProjectPresentation_Jashwanth&Kabhilesh.pptx`](<EECE5640_ProjectPresentation_Jashwanth&Kabhilesh.pptx>) | Original project presentation |
| [`EECE_5640_Jashwanth&Kabhilesh.pdf`](<EECE_5640_Jashwanth&Kabhilesh.pdf>) | PDF version of the project report/presentation |
| `assets/readme/` | Extracted result plots used in this README |

## Reproducing the Wave-U-Net Environment

```bash
cd Wave-U-Net-Pytorch
pip install -r requirements.txt
python train.py --dataset_dir /path/to/MUSDB18HQ
```

Example GPU logging pattern used for the study:

```bash
nvidia-smi --query-gpu=timestamp,pstate,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,power.management --format=csv -l 5
```

## Main Takeaway

For audio deep learning workloads, the V100 is the stronger training GPU when throughput and scaling matter most. The P100 remains useful for lower-power, smaller-batch, edge-style training experiments, but it shows clearer limits as batch size and dataset pressure increase.
