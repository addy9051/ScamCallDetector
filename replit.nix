{pkgs}: {
  deps = [
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.cairo
    pkgs.libsndfile
    pkgs.portaudio
    pkgs.ffmpeg-full
    pkgs.glibcLocales
  ];
}
