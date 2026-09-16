// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: Inria-AIO, Cryspen, and Christian Amsüss
#![no_std]
#![no_main]

use defmt_rtt as _;
use embedded_alloc::LlffHeap as Heap;
use embedded_cal::plumbing::ec::{EcPrimitives, P256};
use embedded_cal_nrf54l15::Nrf54l15Cal;
use libcrux_iot_p256::embedded_cal_integration::{LibcruxEc, LibcruxPoint, LibcruxScalar};
use nrf_pac as _;
use panic_probe as _;
use testvectors::dh::EccVector;

// XXX: `libcrux-iot-p256` still depends on a version of `libcrux-hacl-rs`
// that needs a global allocator.
#[global_allocator]
static HEAP: Heap = Heap::empty();

struct TestState {
    board_cal: Nrf54l15Cal,
}

/// An [`EcPrimitives<P256>`] implementation that mixes two backends: scalar/point encoding
/// is done by libcrux's own software reference implementation ([`LibcruxEc`]), while the
/// actual scalar multiplication is delegated to the nRF54L15's CRACEN hardware accelerator
/// (through the wrapped [`Nrf54l15Cal`]).
///
/// [`Scalar`][EcPrimitives::Scalar] and [`Point`][EcPrimitives::Point] are `LibcruxEc`'s own
/// types; [`multiply_scalar_point`][EcPrimitives::multiply_scalar_point] bridges them to and
/// from `Nrf54l15Cal`'s `NrfScalar`/`NrfPoint` via their shared 32-byte big-endian encoding,
/// since that is the only representation both backends agree on.
struct MixedEc<'a> {
    hw: &'a mut Nrf54l15Cal,
    sw: LibcruxEc,
}

impl<'a> MixedEc<'a> {
    fn new(hw: &'a mut Nrf54l15Cal) -> Self {
        Self { hw, sw: LibcruxEc }
    }
}

impl<'a> EcPrimitives<P256> for MixedEc<'a> {
    const HAS_MULTIPLY_SCALAR_POINT: bool = true;

    type Scalar = LibcruxScalar;
    type Point = LibcruxPoint;

    fn import_scalar_bytes(&mut self, scalar: &[u8]) -> Result<Self::Scalar, embedded_cal::ImportError> {
        self.sw.import_scalar_bytes(scalar)
    }

    fn point(&mut self, x: Self::Scalar, y: Self::Scalar) -> Self::Point {
        self.sw.point(x, y)
    }

    fn export_scalar_bytes<'s>(&mut self, scalar: &'s Self::Scalar) -> impl AsRef<[u8]> + use<'s> {
        self.sw.export_scalar_bytes(scalar)
    }

    fn x_coord(&mut self, point: &Self::Point) -> Self::Scalar {
        self.sw.x_coord(point)
    }

    fn y_coord(&mut self, point: &Self::Point) -> Self::Scalar {
        self.sw.y_coord(point)
    }

    /// Bridges the software `a`/`b` into the hardware backend's own types, performs the
    /// multiplication on CRACEN, then bridges the result back.
    ///
    /// # Panics
    ///
    /// Panics if the hardware's result is not on the curve (via
    /// [`LibcruxEc::point`][EcPrimitives::point]'s validation) -- see that method's docs.
    fn multiply_scalar_point(&mut self, a: &Self::Scalar, b: &Self::Point) -> Self::Point {
        // `Nrf54l15Cal` implements `EcPrimitives<C>` generically for every curve it
        // supports, so calls on `self.hw` need `C = P256` spelled out explicitly --
        // otherwise the compiler has no way to pick which curve's Scalar/Point to produce.
        let a_bytes = self.sw.export_scalar_bytes(a);
        let hw_a = <Nrf54l15Cal as EcPrimitives<P256>>::import_scalar_bytes(self.hw, a_bytes.as_ref())
            .expect("libcrux scalars are always 32 well-formed bytes");

        let bx = self.sw.x_coord(b);
        let bx_bytes = self.sw.export_scalar_bytes(&bx);
        let hw_x = <Nrf54l15Cal as EcPrimitives<P256>>::import_scalar_bytes(self.hw, bx_bytes.as_ref())
            .expect("libcrux coordinates are always 32 well-formed bytes");

        let by = self.sw.y_coord(b);
        let by_bytes = self.sw.export_scalar_bytes(&by);
        let hw_y = <Nrf54l15Cal as EcPrimitives<P256>>::import_scalar_bytes(self.hw, by_bytes.as_ref())
            .expect("libcrux coordinates are always 32 well-formed bytes");

        let hw_point = <Nrf54l15Cal as EcPrimitives<P256>>::point(self.hw, hw_x, hw_y);

        // The actual multiplication: CRACEN hardware.
        let hw_result = <Nrf54l15Cal as EcPrimitives<P256>>::multiply_scalar_point(self.hw, &hw_a, &hw_point);

        let rx = <Nrf54l15Cal as EcPrimitives<P256>>::x_coord(self.hw, &hw_result);
        let rx_bytes = <Nrf54l15Cal as EcPrimitives<P256>>::export_scalar_bytes(self.hw, &rx);
        let sw_x = self
            .sw
            .import_scalar_bytes(rx_bytes.as_ref())
            .expect("Nrf54l15Cal always exports 32 well-formed bytes");

        let ry = <Nrf54l15Cal as EcPrimitives<P256>>::y_coord(self.hw, &hw_result);
        let ry_bytes = <Nrf54l15Cal as EcPrimitives<P256>>::export_scalar_bytes(self.hw, &ry);
        let sw_y = self
            .sw
            .import_scalar_bytes(ry_bytes.as_ref())
            .expect("Nrf54l15Cal always exports 32 well-formed bytes");

        self.sw.point(sw_x, sw_y)
    }
}

#[defmt_test::tests]
mod tests {
    use embedded_cal::plumbing::ec::Ec;
    use testvectors::dh::RFC5903_P256;

    use super::*;

    #[init]
    fn init() -> super::TestState {
        // FIXME: How to make sure there is a exclusive reference for CRACEN_S?
        let board_cal =
            embedded_cal_nrf54l15::Nrf54l15Cal::new(nrf_pac::CRACEN_S, nrf_pac::CRACENCORE_S);

        super::TestState { board_cal }
    }

    #[test]
    fn test_ecc_p256_board(state: &mut super::TestState) {
        let ec = state.board_cal.p256();

        for v in RFC5903_P256 {
            LibcruxTestVector(v).libcrux_test_with(ec);
        }
    }

    #[test]
    fn test_ecc_p256_software(_state: &mut super::TestState) {
        let mut ec = libcrux_iot_p256::embedded_cal_integration::LibcruxEc;
        for v in RFC5903_P256 {
            LibcruxTestVector(v).libcrux_test_with(&mut ec);
        }
    }

    #[test]
    fn test_ecc_p256_mixed(state: &mut super::TestState) {
        let mut ec = super::MixedEc::new(state.board_cal.p256());

        for v in RFC5903_P256 {
            LibcruxTestVector(v).libcrux_test_with(&mut ec);
        }
    }
}

struct LibcruxTestVector<'a>(&'a EccVector);
impl<'a> LibcruxTestVector<'a> {
    /// Runs the test vector by the Cal implementation.
    ///
    /// Panics if either the algorithm is not supported, or either direction of running DH does not
    /// result in the expected shared secret.
    pub fn libcrux_test_with<C: embedded_cal::plumbing::ec::EcPrimitives<P256>>(&self, ec: &mut C) {
        let mut alice_public_computed = [0u8; 64];
        assert!(libcrux_iot_p256::embedded_cal_integration::dh_initiator_ec(
            ec,
            &mut alice_public_computed,
            self.0.alice_private
        ));
        assert_eq!(self.0.alice_public, &alice_public_computed[..32]);

        let mut bob_public_computed = [0u8; 64];
        assert!(libcrux_iot_p256::embedded_cal_integration::dh_initiator_ec(
            ec,
            bob_public_computed.as_mut_slice(),
            self.0.bob_private
        ));
        assert_eq!(self.0.bob_public, &bob_public_computed[..32]);

        let mut alice_shared_secret = [0u8; 64];
        assert!(libcrux_iot_p256::embedded_cal_integration::dh_responder_ec(
            ec,
            alice_shared_secret.as_mut_slice(),
            &bob_public_computed,
            self.0.alice_private,
        ));
        assert_eq!(&alice_shared_secret[..32], self.0.shared_secret);

        let mut bob_shared_secret = [0u8; 64];
        assert!(libcrux_iot_p256::embedded_cal_integration::dh_responder_ec(
            ec,
            bob_shared_secret.as_mut_slice(),
            &alice_public_computed,
            self.0.bob_private,
        ));
        assert_eq!(&bob_shared_secret[..32], self.0.shared_secret);
    }
}
