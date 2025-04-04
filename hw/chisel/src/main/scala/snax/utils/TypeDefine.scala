package snax.utils

import chisel3._

// simplified tcdm interface
class TcdmReq(addrWidth: Int, tcdmDataWidth: Int) extends Bundle {
  val addr  = UInt(addrWidth.W)
  val write = Bool()
  val data  = UInt(tcdmDataWidth.W)
  val strb  = UInt((tcdmDataWidth / 8).W)
}

class TcdmRsp(tcdmDataWidth: Int) extends Bundle {
  val data = UInt(tcdmDataWidth.W)
}

// simplified csr read/write cmd
class CsrReq(addrWidth: Int) extends Bundle {
  val data  = UInt(32.W)
  val addr  = UInt(addrWidth.W)
  val write = Bool()
}

class CsrRsp extends Bundle {
  val data = UInt(32.W)
}
