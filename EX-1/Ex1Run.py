from Grid import Grid
import matplotlib.pyplot as plt
from phantom import phantom


shepp = phantom(n=512, p_type='Modified Shepp-Logan', ellipses=None)

# shepp = utils.shepp_logan(512)
shepp_grid = Grid(512, 512, [0.5, 0.5])
shepp_grid.set_buffer(shepp)

print(shepp_grid.get_at_physical(-90.25, 47.5))
print(shepp_grid.get_at_physical(11.75, -4.0))

print(shepp_grid.index_to_physical(145, 399))
print(shepp_grid.physical_to_index(-0.4, 110.4))

plt.imshow(shepp_grid.buffer)
plt.gray()
plt.show()
