# Control system framework

Therea are several frameworks for robotics applications, with the ROS being the most well known. Another, lesser known example is Drake. Here we create our own framework, not for the sake of creating frameworks (though it is fun), but for making things right.

In its essense, every system is a control loop, that reads some inputs and produces some outputs. Then inputs of one system are connected to the outputs of another, etc. Some of the inputs are connected to the real world (i.e. cameras, sensors, etc.) and some outputs are connected to the robotic actuators.

* One output may be connected to multiple inputs, but every input has just one respective output
* Every connection might be pull or push. In pull connection, the reader reads values from connection when needed. In push connection, the writer pushes data as it pleases, and the reader has to keep up with data
* Every pull connection can be turned into push by polling the data every now and on
* Every push connection can be turned into pull by caching the last value.
* The push connection are done via `Port`s
* The pull connections are done via `Properties`

Systems can live in different `World`s, like threads (when we migrate library to make every control system a coroutine), processes on the same or different machines. Ideally, when we bind inputs to outputs, we want framework to figure out how to connect them together. It can be as simple as shared memory, async or threaded queues, sockets or web-sockets, etc.

It is important to keep the control systems pure and move all the sophisticated logic like caching, polling and similar to the channels. Consider the following examples:
```python
inp_sys.ins.status = CachedProperty(out_sys.outs.status, every_sec=1.)
inp_sys.ins.telemetry = ThrottledChannel(out_sys.outs.telemtry, rps=10)
```