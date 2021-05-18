# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['Tool_ver1.0.py'],
             pathex=['C:\\Users\\82109\\Desktop\\tool_ver1.0'],
             binaries=[],
             datas=[],
             hiddenimports=["['tensorflow','tensorflow.keras']"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Tool_ver1.0',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True , icon='image.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Tool_ver1.0')
