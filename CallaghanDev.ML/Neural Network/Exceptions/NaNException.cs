﻿namespace CallaghanDev.ML.Exceptions
{
    public class NaNException : Exception
    {
        public NaNException() : base("NaN detected") { }

        public NaNException(string message) : base(message) { }

        public NaNException(string message, Exception inner) : base(message, inner) { }
    }

}
