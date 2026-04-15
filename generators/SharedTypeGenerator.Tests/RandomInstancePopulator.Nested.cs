using System.Collections;
using System.Reflection;
using Renderite.Shared;

namespace SharedTypeGenerator.Tests;

public static partial class RandomInstancePopulator
{
    private static object PopulateValueType(Type valueType, Random rng, Assembly assembly, HashSet<object> seen, bool useMinimalValues = false)
    {
        var instance = Activator.CreateInstance(valueType)!;
        PopulateInternal(instance, valueType, rng, assembly, seen, useMinimalValues);
        return instance;
    }

    private static bool IsMemoryPackableClass(Type type)
    {
        return type is { IsClass: true, IsAbstract: false } && typeof(IMemoryPackable).IsAssignableFrom(type);
    }

    private static object CreateAndPopulate(Type type, Random rng, Assembly assembly, HashSet<object> seen, bool useMinimalValues = false)
    {
        var instance = Activator.CreateInstance(type) ?? throw new InvalidOperationException($"Cannot create {type.Name}");
        PopulateInternal(instance, type, rng, assembly, seen, useMinimalValues);
        return instance;
    }

    private static bool IsListOfT(Type type, out Type? elementType)
    {
        elementType = null;
        if (!type.IsGenericType) return false;
        var def = type.GetGenericTypeDefinition();
        if (def != typeof(List<>)) return false;
        elementType = type.GetGenericArguments()[0];
        return true;
    }

    /// <summary>Creates a list with 0–4 random elements. Enum elements use the same underlying-value
    /// distribution as <see cref="GetRandomEnumValue"/>.</summary>
    private static IList CreateRandomList(Type elementType, Random rng, Assembly assembly, HashSet<object> seen)
    {
        var listType = typeof(List<>).MakeGenericType(elementType);
        var list = (IList)Activator.CreateInstance(listType)!;
        var count = rng.Next(0, 4);
        for (int i = 0; i < count; i++)
        {
            object? elem;
            if (elementType.IsValueType && !elementType.IsPrimitive && elementType != typeof(Guid))
                elem = PopulateValueType(elementType, rng, assembly, seen);
            else if (IsMemoryPackableClass(elementType))
                elem = CreateAndPopulate(elementType, rng, assembly, seen);
            else if (elementType == typeof(int) || elementType == typeof(long))
                elem = rng.Next();
            else if (elementType == typeof(float) || elementType == typeof(double))
                elem = rng.NextDouble();
            else if (elementType == typeof(bool))
                elem = rng.Next(2) == 1;
            else if (elementType == typeof(Guid))
                elem = Guid.NewGuid();
            else if (elementType.IsEnum)
                elem = GetRandomEnumValue(elementType, rng);
            else
                continue;
            list.Add(elem!);
        }
        return list;
    }
}
