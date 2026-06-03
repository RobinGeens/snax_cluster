// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: SHL-0.51
//
// Real-core demonstration of the snax_intf_translator CSR-read id-drop bug and
// its software fix. See csr_id_bug_demo/README.md for the full explanation.
//
// The real Snitch core runs the buggy poll `csrr a3,0x495 ; bltu a3,a2` (HANGS)
// and the fixed poll `csrr a3,0x495 ; add a3,a3,zero ; bltu` (COMPLETES) through
// the real taped-out translator + mux + ReqRspManager. The only modeled element
// is rsp_defer: a 1-cycle withhold of the response handshake (the stream_arbiter
// grant defer).

`timescale 1ns/1ps

`include "reqrsp_interface/typedef.svh"
`include "snitch_vm/typedef.svh"

package snitch_csrbug_pkg;
  import snitch_pkg::*;

  localparam int unsigned AddrWidth = 48;
  localparam int unsigned DataWidth = 64;

  typedef logic [AddrWidth-1:0]   addr_t;
  typedef logic [DataWidth-1:0]   data_t;
  typedef logic [DataWidth/8-1:0] strb_t;

  // Data (TCDM) port types -- tied off, no loads/stores in the programs.
  `REQRSP_TYPEDEF_ALL(dmem, addr_t, data_t, strb_t)

  // Page-table types -- VMSupport=0, tied off.
  `SNITCH_VM_TYPEDEF(AddrWidth)

  // Accelerator offload types, bit-identical to the cluster wrapper.
  typedef struct packed {
    snitch_pkg::acc_addr_e addr;
    logic [4:0]            id;
    logic [31:0]           data_op;
    data_t                 data_arga;
    data_t                 data_argb;
    addr_t                 data_argc;
  } acc_req_t;

  typedef struct packed {
    logic [4:0] id;
    logic       error;
    data_t      data;
  } acc_resp_t;
endpackage

// rsp_defer -- withhold the response handshake for N cycles (models the
// stream_arbiter grant defer). Combinational pass-through once elapsed.
module rsp_defer import snitch_csrbug_pkg::*; #(
  parameter int unsigned N = 1
) (
  input  logic      clk_i,
  input  logic      rst_ni,
  // translator side
  input  logic      xlat_pvalid_i,
  input  acc_resp_t xlat_resp_i,
  output logic      xlat_pready_o,
  // core side
  output logic      core_pvalid_o,
  output acc_resp_t core_resp_o,
  input  logic      core_pready_i
);
  logic [7:0] cnt_q;
  logic       withholding;

  assign withholding   = (cnt_q < N);
  assign core_resp_o   = xlat_resp_i;
  assign core_pvalid_o = xlat_pvalid_i & ~withholding;
  assign xlat_pready_o = core_pready_i & ~withholding;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)               cnt_q <= '0;       // armed
    else if (!xlat_pvalid_i)   cnt_q <= '0;       // no response pending -> re-arm
    else if (withholding)      cnt_q <= cnt_q + 1;// counting the defer window
    // else: transparent; stays until xlat_pvalid drops (response consumed)
  end
endmodule

// One real-core chain. FIXED picks the buggy vs the fixed program; the two
// differ only by the inserted `add a3,a3,zero` filler.
module snitch_csr_chain import snitch_csrbug_pkg::*; #(
  parameter bit          FIXED   = 1'b0,
  parameter int unsigned DEFER_N = 1,
  parameter logic [31:0] GAUGE   = 32'hCAFE_F00D
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  output logic [31:0] pc_o,            // current fetch PC (cluster-local)
  output logic [4:0]  live_id_o,       // acc_qreq_o.id = decode-stage rd
  output logic        acc_qvalid_o,    // core is offloading this cycle
  output logic        consume_o,       // response consumed by the core this cycle
  output logic [4:0]  delivered_id_o,  // id the response carries when consumed
  output logic [31:0] delivered_data_o
);

  // ---- instruction ROM ----------------------------------------------------
  // Word-addressed program at PC 0. j-self loops mark "done".
  //   a2 = x12 (compare bound = 1), a3 = x13 (poll register)
  localparam logic [31:0] I_ADDI_A2_1  = 32'h00100613; // addi a2,x0,1
  localparam logic [31:0] I_ADDI_A3_0  = 32'h00000693; // addi a3,x0,0
  localparam logic [31:0] I_CSRR_A3    = 32'h495026f3; // csrr a3,0x495  (csrrs a3,0x495,x0)
  localparam logic [31:0] I_ADD_A3     = 32'h000686b3; // add  a3,a3,zero (the safe filler)
  localparam logic [31:0] I_BLTU_M4    = 32'hfec6eee3; // bltu a3,a2,-4   (buggy: loop = csrr@0x08)
  localparam logic [31:0] I_BLTU_M8    = 32'hfec6ece3; // bltu a3,a2,-8   (fixed: loop = csrr@0x08)
  localparam logic [31:0] I_JSELF      = 32'h0000006f; // jal x0,0        (j .)

  logic [31:0] rom [0:31];
  // DONE PC: the j-self the program falls through to once a3 >= a2.
  localparam logic [31:0] DONE_PC = FIXED ? 32'h14 : 32'h10;

  initial begin
    for (int i = 0; i < 32; i++) rom[i] = I_JSELF; // default: spin
    rom[0] = I_ADDI_A2_1;   // 0x00
    rom[1] = I_ADDI_A3_0;   // 0x04
    rom[2] = I_CSRR_A3;     // 0x08  <- loop target
    if (FIXED) begin
      rom[3] = I_ADD_A3;    // 0x0c  filler (rd=a3, reads a3 -> stalls, holds bus id)
      rom[4] = I_BLTU_M8;   // 0x10  bltu a3,a2,0x08
      rom[5] = I_JSELF;     // 0x14  done
    end else begin
      rom[3] = I_BLTU_M4;   // 0x0c  bltu a3,a2,0x08
      rom[4] = I_JSELF;     // 0x10  done
    end
  end

  // ---- snitch core boundary signals ---------------------------------------
  addr_t       inst_addr;
  logic        inst_valid, inst_ready, inst_cacheable;
  logic [31:0] inst_data;

  acc_req_t    acc_qreq;
  logic        acc_qvalid, acc_qready;
  acc_resp_t   acc_prsp;
  logic        acc_pvalid, acc_pready;

  dmem_req_t   data_req;
  dmem_rsp_t   data_rsp;

  // unused fp / ptw / misc
  logic            flush_i_valid;

  assign inst_ready  = 1'b1;
  assign inst_data   = rom[inst_addr[6:2]];
  assign pc_o        = inst_addr[31:0];

  // No loads/stores are issued; keep the data port permanently accepting.
  assign data_rsp = '{p: '0, p_valid: 1'b0, q_ready: 1'b1};

  snitch #(
    .AddrWidth              (AddrWidth),
    .DataWidth              (DataWidth),
    .RVE                    (1'b0),
    .Xdma                   (1'b0),
    .Xssr                   (1'b0),
    .FP_EN                  (1'b0),
    .RVF                    (1'b0),
    .RVD                    (1'b0),
    .XF16                   (1'b0),
    .XF16ALT                (1'b0),
    .XF8                    (1'b0),
    .XF8ALT                 (1'b0),
    .XDivSqrt               (1'b0),
    .XFVEC                  (1'b0),
    .XFDOTP                 (1'b0),
    .XFAUX                  (1'b0),
    .FLEN                   (DataWidth),
    .VMSupport              (1'b0),
    .Xipu                   (1'b0),
    .dreq_t                 (dmem_req_t),
    .drsp_t                 (dmem_rsp_t),
    .acc_req_t              (acc_req_t),
    .acc_resp_t             (acc_resp_t),
    .pa_t                   (pa_t),
    .l0_pte_t               (l0_pte_t),
    .NumIntOutstandingLoads (1),
    .NumIntOutstandingMem   (4),
    .NumDTLBEntries         (1),
    .NumITLBEntries         (1),
    .SnitchPMACfg           ('{default: 0}),
    .DebugSupport           (1'b0)
  ) i_snitch (
    .clk_i               (clk_i),
    .rst_i               (~rst_ni),
    .hart_id_i           (32'd0),
    .cluster_core_id_i   (32'd0),
    .irq_i               ('0),
    .boot_addr_i         (32'd0),
    .cluster_base_addr_i ('0),
    .flush_i_valid_o     (flush_i_valid),
    .flush_i_ready_i     (1'b1),
    .inst_addr_o         (inst_addr),
    .inst_cacheable_o    (inst_cacheable),
    .inst_data_i         (inst_data),
    .inst_valid_o        (inst_valid),
    .inst_ready_i        (inst_ready),
    .acc_qreq_o          (acc_qreq),
    .acc_qvalid_o        (acc_qvalid),
    .acc_qready_i        (acc_qready),
    .acc_prsp_i          (acc_prsp),
    .acc_pvalid_i        (acc_pvalid),
    .acc_pready_o        (acc_pready),
    .data_req_o          (data_req),
    .data_rsp_i          (data_rsp),
    .ptw_valid_o         (),
    .ptw_ready_i         ('0),
    .ptw_va_o            (),
    .ptw_ppn_o           (),
    .ptw_pte_i           ('0),
    .ptw_is_4mega_i      ('0),
    .fpu_rnd_mode_o      (),
    .fpu_fmt_mode_o      (),
    .fpu_status_i        ('0),
    .core_events_o       (),
    .obs_o               (),
    .multi_acc_mux_o     (),
    .snax_barrier_i      (1'b0),
    .barrier_o           (),
    .barrier_i           (1'b0)
  );

  assign live_id_o     = acc_qreq.id;
  assign acc_qvalid_o  = acc_qvalid;

  // ---- REAL taped-out translator ------------------------------------------
  logic [31:0] t_req_data, t_req_addr;
  logic        t_req_wen, t_req_valid, t_req_ready;
  logic [31:0] t_rsp_data;
  logic        t_rsp_valid, t_rsp_ready;

  acc_resp_t xlat_resp;
  logic      xlat_pvalid, xlat_pready;

  // The restored taped-out (buggy) translator -- the logic the chip shipped with.
  snax_intf_translator_orig #(
    .acc_req_t (acc_req_t), .acc_rsp_t (acc_resp_t),
    .NumOutstandingLoads (4), .CsrAddrOffset (32'h3c0)
  ) i_xlat (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .snax_req_i (acc_qreq), .snax_qvalid_i (acc_qvalid), .snax_qready_o (acc_qready),
    .snax_resp_o (xlat_resp), .snax_pvalid_o (xlat_pvalid), .snax_pready_i (xlat_pready),
    .snax_csr_req_bits_data_o (t_req_data), .snax_csr_req_bits_addr_o (t_req_addr),
    .snax_csr_req_bits_write_o (t_req_wen), .snax_csr_req_valid_o (t_req_valid),
    .snax_csr_req_ready_i (t_req_ready),
    .snax_csr_rsp_bits_data_i (t_rsp_data), .snax_csr_rsp_valid_i (t_rsp_valid),
    .snax_csr_rsp_ready_o (t_rsp_ready)
  );

  // ---- REAL mux_demux + ReqRspManager (csrman) ----------------------------
  logic [1:0][31:0] mx_req_addr, mx_req_data, mx_rsp_data;
  logic [1:0][0:0]  mx_req_wen, mx_req_valid, mx_req_ready, mx_rsp_valid, mx_rsp_ready;
  logic        cs_req_ready, cs_rsp_valid;
  logic [31:0] cs_rsp_data;

  snax_csr_mux_demux #(
    .AddrSelOffSet (204), .RegDataWidth (32), .RegAddrWidth (32)
  ) i_mux (
    .csr_req_addr_i (t_req_addr), .csr_req_data_i (t_req_data), .csr_req_wen_i (t_req_wen),
    .csr_req_valid_i (t_req_valid), .csr_req_ready_o (t_req_ready),
    .csr_rsp_data_o (t_rsp_data), .csr_rsp_valid_o (t_rsp_valid), .csr_rsp_ready_i (t_rsp_ready),
    .acc_csr_req_addr_o (mx_req_addr), .acc_csr_req_data_o (mx_req_data),
    .acc_csr_req_wen_o (mx_req_wen), .acc_csr_req_valid_o (mx_req_valid),
    .acc_csr_req_ready_i (mx_req_ready),
    .acc_csr_rsp_data_i (mx_rsp_data), .acc_csr_rsp_valid_i (mx_rsp_valid),
    .acc_csr_rsp_ready_o (mx_rsp_ready)
  );

  assign mx_req_ready = {cs_req_ready, 1'b0};
  assign mx_rsp_valid = {cs_rsp_valid, 1'b0};
  assign mx_rsp_data  = {cs_rsp_data,  32'h0};

  snax_simbacore_reqrspman_ReqRspManager i_csrman (
    .clock (clk_i), .reset (~rst_ni),
    .io_reqRspIO_req_ready (cs_req_ready),
    .io_reqRspIO_req_valid (mx_req_valid[1]),
    .io_reqRspIO_req_bits_addr (mx_req_addr[1]),
    .io_reqRspIO_req_bits_write (mx_req_wen[1]),
    .io_reqRspIO_req_bits_data (mx_req_data[1]),
    .io_reqRspIO_req_bits_strb (4'hf),
    .io_reqRspIO_req_bits_priority (1'b0),
    .io_reqRspIO_rsp_ready (mx_rsp_ready[1]),
    .io_reqRspIO_rsp_valid (cs_rsp_valid),
    .io_reqRspIO_rsp_bits_data (cs_rsp_data),
    .io_readWriteRegIO_ready (1'b1),
    .io_readWriteRegIO_valid (),
    .io_readWriteRegIO_bits_0 (), .io_readWriteRegIO_bits_1 (), .io_readWriteRegIO_bits_2 (),
    .io_readWriteRegIO_bits_3 (), .io_readWriteRegIO_bits_4 (), .io_readWriteRegIO_bits_5 (),
    .io_readWriteRegIO_bits_6 (),
    .io_readOnlyReg_0 (32'h0), .io_readOnlyReg_1 (32'h0),
    .io_readOnlyReg_2 (GAUGE),  // R10_DELAY_GAUGE (csrman addr 9 via mux offset 204)
    .io_readOnlyReg_3 (32'h0)
  );

  // ---- modeled 1-cycle arbiter withhold on the response path --------------
  rsp_defer #(.N (DEFER_N)) i_defer (
    .clk_i (clk_i), .rst_ni (rst_ni),
    .xlat_pvalid_i (xlat_pvalid), .xlat_resp_i (xlat_resp), .xlat_pready_o (xlat_pready),
    .core_pvalid_o (acc_pvalid), .core_resp_o (acc_prsp), .core_pready_i (acc_pready)
  );

  assign consume_o        = acc_pvalid & acc_pready;
  assign delivered_id_o   = acc_prsp.id;
  assign delivered_data_o = acc_prsp.data[31:0];
endmodule

// ---------------------------------------------------------------------------
// Testbench: run the buggy and fixed chains side by side on real cores.
// ---------------------------------------------------------------------------
module tb_snitch_csr_id_bug;
  import snitch_csrbug_pkg::*;

  localparam logic [31:0] GAUGE  = 32'hCAFE_F00D;
  localparam logic [4:0]  A3_ID  = 5'd13;
  localparam int unsigned RUN_CYCLES = 400;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  always #5 clk = ~clk;

  // buggy chain
  logic [31:0] b_pc, b_ddata;  logic [4:0] b_live, b_did;
  logic        b_qv, b_consume;
  // fixed chain
  logic [31:0] f_pc, f_ddata;  logic [4:0] f_live, f_did;
  logic        f_qv, f_consume;

  snitch_csr_chain #(.FIXED(1'b0), .DEFER_N(1), .GAUGE(GAUGE)) c_buggy (
    .clk_i(clk), .rst_ni(rst_n),
    .pc_o(b_pc), .live_id_o(b_live), .acc_qvalid_o(b_qv),
    .consume_o(b_consume), .delivered_id_o(b_did), .delivered_data_o(b_ddata));

  snitch_csr_chain #(.FIXED(1'b1), .DEFER_N(1), .GAUGE(GAUGE)) c_fixed (
    .clk_i(clk), .rst_ni(rst_n),
    .pc_o(f_pc), .live_id_o(f_live), .acc_qvalid_o(f_qv),
    .consume_o(f_consume), .delivered_id_o(f_did), .delivered_data_o(f_ddata));

  localparam logic [31:0] BUGGY_DONE = 32'h10;
  localparam logic [31:0] FIXED_DONE = 32'h14;
  localparam logic [31:0] BUGGY_BLTU = 32'h0c; // where the buggy poll wedges

  // observe
  logic        b_reached_done, f_reached_done;
  logic [4:0]  b_first_did, f_first_did;
  logic [31:0] b_first_data, f_first_data;
  logic        b_seen_consume, f_seen_consume;
  int          b_stuck_cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      b_reached_done <= 1'b0; f_reached_done <= 1'b0;
      b_seen_consume <= 1'b0; f_seen_consume <= 1'b0;
      b_first_did <= '0;      f_first_did <= '0;
      b_first_data <= '0;     f_first_data <= '0;
      b_stuck_cnt <= 0;
    end else begin
      if (b_pc == BUGGY_DONE) b_reached_done <= 1'b1;
      if (f_pc == FIXED_DONE) f_reached_done <= 1'b1;
      if (b_consume && !b_seen_consume) begin b_seen_consume <= 1'b1; b_first_did <= b_did; b_first_data <= b_ddata; end
      if (f_consume && !f_seen_consume) begin f_seen_consume <= 1'b1; f_first_did <= f_did; f_first_data <= f_ddata; end
      if (b_pc == BUGGY_BLTU) b_stuck_cnt <= b_stuck_cnt + 1;
    end
  end

  int errors = 0;

  initial begin
    rst_n = 1'b0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    $display("======================================================================");
    $display(" CSR-read id-drop bug on the REAL Snitch core (see README.md)");
    $display(" gauge = 0x%08h, poll register a3 = x%0d", GAUGE, A3_ID);
    $display(" buggy : csrr a3,0x495 ; bltu a3,a2");
    $display(" fixed : csrr a3,0x495 ; add a3,a3,zero ; bltu a3,a2");
    $display("======================================================================");

    // per-cycle trace
    $display("");
    $display(" cyc | BUGGY pc liveid qv cons did data       | FIXED pc liveid qv cons did data");
    for (int k = 0; k < 14; k++) begin
      @(posedge clk);
      $display(" %3d | 0x%02h x%0d %0b %0b x%0d 0x%08h | 0x%02h x%0d %0b %0b x%0d 0x%08h",
        k, b_pc, b_live, b_qv, b_consume, b_did, b_ddata,
           f_pc, f_live, f_qv, f_consume, f_did, f_ddata);
    end

    repeat (RUN_CYCLES) @(posedge clk);

    $display("");
    $display(" BUGGY: final PC=0x%02h, gauge consumed at id=x%0d (should be a3=x%0d), wedged %0d cyc",
             b_pc, b_first_did, A3_ID, b_stuck_cnt);
    if (!b_reached_done && b_seen_consume && (b_first_did != A3_ID))
      $display("   -> BUG REPRODUCED: gauge landed on x%0d, a3 never written -> poll HANGS", b_first_did);
    else begin
      errors++;
      $display("   !! buggy chain did NOT hang as expected (reached_done=%0b, did=x%0d)",
               b_reached_done, b_first_did);
    end

    $display(" FIXED: final PC=0x%02h, gauge consumed at id=x%0d  data=0x%08h",
             f_pc, f_first_did, f_first_data);
    if (f_reached_done && f_seen_consume && (f_first_did == A3_ID) && (f_first_data == GAUGE))
      $display("   -> SW FIX WORKS: filler holds id=a3, gauge delivered to a3 -> poll completes");
    else begin
      errors++;
      $display("   !! fixed chain failed (reached_done=%0b, did=x%0d, data=0x%08h)",
               f_reached_done, f_first_did, f_first_data);
    end

    $display("");
    if (errors == 0)
      $display(" RESULT: PASS  -- buggy poll hangs, read_snax_csr_safe filler completes (real core)");
    else
      $display(" RESULT: FAIL  -- %0d unexpected outcome(s)", errors);
    $finish;
  end

  // watchdog
  initial begin
    #200000;
    $display(" RESULT: FAIL -- watchdog timeout");
    $finish;
  end
endmodule
