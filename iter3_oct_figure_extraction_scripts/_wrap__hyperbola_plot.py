import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64, io, json, sys, traceback, os

_FIGS = []
_ORIG_SHOW = plt.show
def _capture_show(*a, **k):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
        buf.seek(0)
        _FIGS.append(base64.b64encode(buf.read()).decode('ascii'))
    plt.close('all')
plt.show = _capture_show

try:
    _ns = {'__name__': '__main__', '__file__': '/home/govindu.srimaan/project/_hyperbola_plot.py'}
    exec(compile(open('/home/govindu.srimaan/project/_hyperbola_plot.py').read(), '/home/govindu.srimaan/project/_hyperbola_plot.py', 'exec'), _ns)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()

_capture_show()
print('__FIGURES__' + json.dumps(_FIGS))
