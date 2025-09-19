from InterpolatorInterface import InterpolatorInterface
interpolator = InterpolatorInterface()
result = interpolator.generate(
    imgs=('img0.png', 'img1.png'),
    exp=3,
    outputdir="interpolator_out"
)
print(result)
result = interpolator.generate(
    imgs=('img0.png', 'img1.png'),
    exp=3
)
print(result)
