// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use egui::{Key, Modifiers};
use winit::event::{ModifiersState, VirtualKeyCode};

/// Utility struct to convert egui items to winit
pub struct EguiToWinit;

impl EguiToWinit {
    /// Converts [`egui::CursorIcon`] to [`winit::window::CursorIcon`].
    pub(crate) fn cursor(cursor: egui::CursorIcon) -> winit::window::CursorIcon {
        match cursor {
            egui::CursorIcon::Default => winit::window::CursorIcon::Default,
            egui::CursorIcon::PointingHand => winit::window::CursorIcon::Hand,
            egui::CursorIcon::ResizeHorizontal => winit::window::CursorIcon::ColResize,
            egui::CursorIcon::ResizeNeSw => winit::window::CursorIcon::NeResize,
            egui::CursorIcon::ResizeNwSe => winit::window::CursorIcon::NwResize,
            egui::CursorIcon::ResizeVertical => winit::window::CursorIcon::RowResize,
            egui::CursorIcon::Text => winit::window::CursorIcon::Text,
            egui::CursorIcon::Grab => winit::window::CursorIcon::Grab,
            egui::CursorIcon::Grabbing => winit::window::CursorIcon::Grabbing,
            egui::CursorIcon::NotAllowed => winit::window::CursorIcon::NotAllowed,
            egui::CursorIcon::ZoomIn => winit::window::CursorIcon::ZoomIn,
            egui::CursorIcon::ZoomOut => winit::window::CursorIcon::ZoomOut,
            egui::CursorIcon::AllScroll => winit::window::CursorIcon::AllScroll,
            egui::CursorIcon::NoDrop => winit::window::CursorIcon::NoDrop,
            egui::CursorIcon::Move => winit::window::CursorIcon::Move,
            egui::CursorIcon::Copy => winit::window::CursorIcon::Copy,
            egui::CursorIcon::Alias => winit::window::CursorIcon::Alias,
            egui::CursorIcon::VerticalText => winit::window::CursorIcon::VerticalText,
            egui::CursorIcon::Crosshair => winit::window::CursorIcon::Crosshair,
            egui::CursorIcon::Cell => winit::window::CursorIcon::Cell,
            egui::CursorIcon::Wait => winit::window::CursorIcon::Wait,
            egui::CursorIcon::Progress => winit::window::CursorIcon::Progress,
            egui::CursorIcon::Help => winit::window::CursorIcon::Help,
            egui::CursorIcon::ContextMenu => winit::window::CursorIcon::ContextMenu,
            egui::CursorIcon::None => winit::window::CursorIcon::Default,
        }
    }
}

/// Utility struct to convert winit items to egui
pub struct WinitToEgui;

impl WinitToEgui {
    /// Converts [`winit::event::VirtualKeyCode`] to [`egui::Key`].
    pub(crate) fn key_code(key: VirtualKeyCode) -> Option<Key> {
        use VirtualKeyCode::*;

        Some(match key {
            Down => Key::ArrowDown,
            Left => Key::ArrowLeft,
            Right => Key::ArrowRight,
            Up => Key::ArrowUp,

            Escape => Key::Escape,
            Tab => Key::Tab,
            Back => Key::Backspace,
            Return => Key::Enter,
            Space => Key::Space,

            Insert => Key::Insert,
            Delete => Key::Delete,
            Home => Key::Home,
            End => Key::End,
            PageUp => Key::PageUp,
            PageDown => Key::PageDown,

            Key0 => Key::Num0,
            Key1 => Key::Num1,
            Key2 => Key::Num2,
            Key3 => Key::Num3,
            Key4 => Key::Num4,
            Key5 => Key::Num5,
            Key6 => Key::Num6,
            Key7 => Key::Num7,
            Key8 => Key::Num8,
            Key9 => Key::Num9,

            A => Key::A,
            B => Key::B,
            C => Key::C,
            D => Key::D,
            E => Key::E,
            F => Key::F,
            G => Key::G,
            H => Key::H,
            I => Key::I,
            J => Key::J,
            K => Key::K,
            L => Key::L,
            M => Key::M,
            N => Key::N,
            O => Key::O,
            P => Key::P,
            Q => Key::Q,
            R => Key::R,
            S => Key::S,
            T => Key::T,
            U => Key::U,
            V => Key::V,
            W => Key::W,
            X => Key::X,
            Y => Key::Y,
            Z => Key::Z,
            _ => return None,
        })
    }

    /// Converts [`winit::event::ModifiersState`] to [`egui::Modifiers`].
    pub(crate) fn modifiers(modifiers: ModifiersState) -> Modifiers {
        Modifiers {
            alt: modifiers.alt(),
            ctrl: modifiers.ctrl(),
            shift: modifiers.shift(),
            #[cfg(target_os = "macos")]
            mac_cmd: modifiers.logo(),
            #[cfg(target_os = "macos")]
            command: modifiers.logo(),
            #[cfg(not(target_os = "macos"))]
            mac_cmd: false,
            #[cfg(not(target_os = "macos"))]
            command: modifiers.ctrl(),
        }
    }

    /// Converts [`winit::event::MouseButton`] to [`egui::PointerButton`].
    pub(crate) fn mouse_button(button: winit::event::MouseButton) -> Option<egui::PointerButton> {
        Some(match button {
            winit::event::MouseButton::Left => egui::PointerButton::Primary,
            winit::event::MouseButton::Right => egui::PointerButton::Secondary,
            winit::event::MouseButton::Middle => egui::PointerButton::Middle,
            _ => return None,
        })
    }
}
